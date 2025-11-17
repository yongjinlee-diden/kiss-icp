// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "VoxelHashMap.hpp"
#include "VoxelUtils.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

// Point-to-point correspondences
using Correspondences = tbb::concurrent_vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>;
// Point-to-plane correspondences: (source_point, target_point, target_normal, combined_consistency)
// Combined consistency = source_consistency * target_consistency (both points must be high quality)
using CorrespondenceWithNormal = std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, double>;
using CorrespondencesWithNormals = tbb::concurrent_vector<CorrespondenceWithNormal>;
using LinearSystem = std::pair<Eigen::Matrix6d, Eigen::Vector6d>;

namespace {
inline double square(double x) { return x * x; }

void TransformPoints(const Sophus::SE3d &T, std::vector<kiss_icp::PointWithNormal> &points) {
    const Eigen::Matrix3d &R = T.rotationMatrix();
    const Eigen::Vector3d &t = T.translation();

    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) {
                       kiss_icp::PointWithNormal transformed;
                       // Transform position (x, y, z)
                       transformed.template head<3>() = R * point.template head<3>() + t;
                       // Transform normal (nx, ny, nz) - rotation only
                       transformed.template segment<3>(3) = R * point.template segment<3>(3);
                       // Preserve consistency
                       transformed(6) = point(6);  // consistency
                       return transformed;
                   });
}

// Point-to-point data association (no normal check)
Correspondences DataAssociation(const std::vector<kiss_icp::PointWithNormal> &points,
                                const kiss_icp::VoxelHashMap &voxel_map,
                                const double max_correspondance_distance) {
    using points_iterator = std::vector<kiss_icp::PointWithNormal>::const_iterator;
    Correspondences correspondences;
    correspondences.reserve(points.size());
    tbb::parallel_for(
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(point);
                if (distance < max_correspondance_distance) {
                    Eigen::Vector3d point_3d = point.template head<3>();
                    Eigen::Vector3d neighbor_3d = closest_neighbor.template head<3>();
                    correspondences.emplace_back(point_3d, neighbor_3d);
                }
            });
        });
    return correspondences;
}

// Point-to-plane data association with normal consistency check (inspired by nv_liom)
CorrespondencesWithNormals DataAssociationWithNormals(
    const std::vector<kiss_icp::PointWithNormal> &points,
    const kiss_icp::VoxelHashMap &voxel_map,
    const double max_correspondance_distance,
    const double normal_consistency_threshold) {

    using points_iterator = std::vector<kiss_icp::PointWithNormal>::const_iterator;
    CorrespondencesWithNormals correspondences;
    correspondences.reserve(points.size());

    tbb::parallel_for(
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        [&](const tbb::blocked_range<points_iterator> &r) {
            std::for_each(r.begin(), r.end(), [&](const auto &point) {
                const auto &[closest_neighbor, distance] = voxel_map.GetClosestNeighbor(point);

                if (distance < max_correspondance_distance) {
                    // Extract normals (segment<3>(3) = nx, ny, nz)
                    Eigen::Vector3d source_normal = point.template segment<3>(3);
                    Eigen::Vector3d target_normal = closest_neighbor.template segment<3>(3);

                    // Normal consistency check (inspired by nv_liom)
                    // Only accept correspondences where normals are similar (within ~10 degrees)
                    double dot_product = source_normal.dot(target_normal);

                    if (dot_product >= normal_consistency_threshold) {
                        Eigen::Vector3d point_3d = point.template head<3>();
                        Eigen::Vector3d neighbor_3d = closest_neighbor.template head<3>();

                        // Compute combined consistency (both source and target quality matter)
                        double source_consistency = point(6);
                        double target_consistency = closest_neighbor(6);
                        double combined_consistency = source_consistency * target_consistency;

                        correspondences.emplace_back(point_3d, neighbor_3d, target_normal, combined_consistency);
                    }
                }
            });
        });

    return correspondences;
}

// Build linear system for point-to-point ICP
LinearSystem BuildLinearSystem(const Correspondences &correspondences, const double kernel_scale) {
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto &[source, target] = correspondence;
        const Eigen::Vector3d residual = source - target;
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source);
        return std::make_tuple(J_r, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return square(kernel_scale) / square(kernel_scale + residual2);
    };

    using correspondence_iterator = Correspondences::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(correspondence);
                    const double w = GM_weight(residual.squaredNorm());
                    return LinearSystem(J_r.transpose() * w * J_r,        // JTJ
                                        J_r.transpose() * w * residual);  // JTr
                });
        },
        sum_linear_systems);

    return {JTJ, JTr};
}

// Build linear system for point-to-plane ICP (nv_liom approach)
LinearSystem BuildLinearSystemPointToPlane(const CorrespondencesWithNormals &correspondences,
                                          const double kernel_scale,
                                          const bool use_consistency_weighting) {
    // Point-to-plane residual: r = n · (p_source - p_target)
    // where n is the target normal vector
    auto compute_jacobian_and_residual = [](const auto &correspondence) {
        const auto &[source, target, normal, consistency] = correspondence;

        // Point-to-plane residual
        const double residual = normal.dot(source - target);

        // Jacobian for point-to-plane ICP
        // dr/dξ where ξ = [ρ, φ] (translation, rotation)
        // dr/dρ = n^T
        // dr/dφ = n^T * [source]_×
        Eigen::Matrix<double, 1, 6> J_r;
        J_r.block<1, 3>(0, 0) = normal.transpose();  // Translation part
        J_r.block<1, 3>(0, 3) = (normal.transpose() * (-Sophus::SO3d::hat(source)));  // Rotation part

        return std::make_tuple(J_r, residual, consistency);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto GM_weight = [&](const double &residual2) {
        return square(kernel_scale) / square(kernel_scale + residual2);
    };

    using correspondence_iterator = CorrespondencesWithNormals::const_iterator;
    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        tbb::blocked_range<correspondence_iterator>{correspondences.cbegin(),
                                                    correspondences.cend()},
        LinearSystem(Eigen::Matrix6d::Zero(), Eigen::Vector6d::Zero()),
        [&](const tbb::blocked_range<correspondence_iterator> &r, LinearSystem J) -> LinearSystem {
            return std::transform_reduce(
                r.begin(), r.end(), J, sum_linear_systems, [&](const auto &correspondence) {
                    const auto &[J_r, residual, consistency] = compute_jacobian_and_residual(correspondence);

                    // Combined weighting: GM_weight * consistency_weight
                    const double gm_w = GM_weight(residual * residual);
                    const double consistency_w = use_consistency_weighting ? consistency : 1.0;
                    const double w = gm_w * consistency_w;

                    return LinearSystem(J_r.transpose() * w * J_r,        // JTJ
                                      J_r.transpose() * w * residual);  // JTr
                });
        },
        sum_linear_systems);

    return {JTJ, JTr};
}

}  // namespace

namespace kiss_icp {

Registration::Registration(int max_num_iteration, double convergence_criterion, int max_num_threads,
                          bool use_normals, double normal_consistency_threshold,
                          bool use_consistency_weighting)
    : max_num_iterations_(max_num_iteration),
      convergence_criterion_(convergence_criterion),
      // Only manipulate the number of threads if the user specifies something greater than 0
      max_num_threads_(max_num_threads > 0 ? max_num_threads
                                           : tbb::this_task_arena::max_concurrency()),
      use_normals_(use_normals),
      normal_consistency_threshold_(normal_consistency_threshold),
      use_consistency_weighting_(use_consistency_weighting) {
    // This global variable requires static duration storage to be able to manipulate the max
    // concurrency from TBB across the entire class
    static const auto tbb_control_settings = tbb::global_control(
        tbb::global_control::max_allowed_parallelism, static_cast<size_t>(max_num_threads_));
}

Sophus::SE3d Registration::AlignPointsToMap(const std::vector<PointWithNormal> &frame,
                                            const VoxelHashMap &voxel_map,
                                            const Sophus::SE3d &initial_guess,
                                            const double max_distance,
                                            const double kernel_scale) {
    if (voxel_map.Empty()) return initial_guess;

    // Transform source points to initial guess
    std::vector<PointWithNormal> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < max_num_iterations_; ++j) {
        if (use_normals_) {
            // Point-to-plane ICP with normal consistency check (nv_liom approach)
            const auto correspondences = DataAssociationWithNormals(
                source, voxel_map, max_distance, normal_consistency_threshold_);
            const auto &[JTJ, JTr] = BuildLinearSystemPointToPlane(correspondences, kernel_scale, use_consistency_weighting_);
            const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
            const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
            TransformPoints(estimation, source);
            T_icp = estimation * T_icp;
            if (dx.norm() < convergence_criterion_) break;
        } else {
            // Point-to-point ICP (original KISS-ICP)
            const auto correspondences = DataAssociation(source, voxel_map, max_distance);
            const auto &[JTJ, JTr] = BuildLinearSystem(correspondences, kernel_scale);
            const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
            const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
            TransformPoints(estimation, source);
            T_icp = estimation * T_icp;
            if (dx.norm() < convergence_criterion_) break;
        }
    }

    return T_icp * initial_guess;
}

}  // namespace kiss_icp
