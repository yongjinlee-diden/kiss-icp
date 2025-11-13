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
#include "VoxelHashMap.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelUtils.hpp"

namespace {
using kiss_icp::Voxel;
static const std::array<Voxel, 27> voxel_shifts{
    {Voxel{0, 0, 0},   Voxel{1, 0, 0},   Voxel{-1, 0, 0},  Voxel{0, 1, 0},   Voxel{0, -1, 0},
     Voxel{0, 0, 1},   Voxel{0, 0, -1},  Voxel{1, 1, 0},   Voxel{1, -1, 0},  Voxel{-1, 1, 0},
     Voxel{-1, -1, 0}, Voxel{1, 0, 1},   Voxel{1, 0, -1},  Voxel{-1, 0, 1},  Voxel{-1, 0, -1},
     Voxel{0, 1, 1},   Voxel{0, 1, -1},  Voxel{0, -1, 1},  Voxel{0, -1, -1}, Voxel{1, 1, 1},
     Voxel{1, 1, -1},  Voxel{1, -1, 1},  Voxel{1, -1, -1}, Voxel{-1, 1, 1},  Voxel{-1, 1, -1},
     Voxel{-1, -1, 1}, Voxel{-1, -1, -1}}};
}  // namespace

namespace kiss_icp {

std::tuple<PointWithNormal, double> VoxelHashMap::GetClosestNeighbor(
    const PointWithNormal &query) const {
    // Convert the point to voxel coordinates (using first 3 components: x, y, z)
    Eigen::Vector3d query_xyz;
    query_xyz << query(0), query(1), query(2);
    const auto &voxel = PointToVoxel(query_xyz, voxel_size_);
    // Find the nearest neighbor
    PointWithNormal closest_neighbor = PointWithNormal::Zero();
    double closest_distance = std::numeric_limits<double>::max();
    std::for_each(voxel_shifts.cbegin(), voxel_shifts.cend(), [&](const auto &voxel_shift) {
        const auto &query_voxel = voxel + voxel_shift;
        auto search = map_.find(query_voxel);
        if (search != map_.end()) {
            const auto &points = search.value();
            const PointWithNormal &neighbor = *std::min_element(
                points.cbegin(), points.cend(), [&](const auto &lhs, const auto &rhs) {
                    return (lhs.template head<3>() - query.template head<3>()).norm() <
                           (rhs.template head<3>() - query.template head<3>()).norm();
                });
            double distance = (neighbor.template head<3>() - query.template head<3>()).norm();
            if (distance < closest_distance) {
                closest_neighbor = neighbor;
                closest_distance = distance;
            }
        }
    });
    return std::make_tuple(closest_neighbor, closest_distance);
}

std::vector<PointWithNormal> VoxelHashMap::Pointcloud() const {
    std::vector<PointWithNormal> points;
    points.reserve(map_.size() * static_cast<size_t>(max_points_per_voxel_));
    std::for_each(map_.cbegin(), map_.cend(), [&](const auto &map_element) {
        const auto &voxel_points = map_element.second;
        points.insert(points.end(), voxel_points.cbegin(), voxel_points.cend());
    });
    points.shrink_to_fit();
    return points;
}

void VoxelHashMap::Update(const std::vector<PointWithNormal> &points,
                          const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const std::vector<PointWithNormal> &points, const Sophus::SE3d &pose) {
    std::vector<PointWithNormal> points_transformed(points.size());
    const Eigen::Matrix3d &R = pose.rotationMatrix();
    const Eigen::Vector3d &t = pose.translation();

    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) {
                       PointWithNormal transformed;
                       // Transform position (x, y, z)
                       transformed.template head<3>() = R * point.template head<3>() + t;
                       // Preserve time
                       transformed(3) = point(3);
                       // Transform normal (nx, ny, nz) - rotation only
                       transformed.template segment<3>(4) = R * point.template segment<3>(4);
                       // Preserve confidence
                       transformed(7) = point(7);  // confidence
                       return transformed;
                   });

    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<PointWithNormal> &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        // Use only x, y, z for voxel computation
        Eigen::Vector3d point_xyz;
        point_xyz << point(0), point(1), point(2);
        const auto voxel = PointToVoxel(point_xyz, voxel_size_);

        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_points = search.value();
            double new_confidence = point(7);  // Extract confidence from new point

            if (voxel_points.size() >= max_points_per_voxel_) {
                // Confidence-based quality control: replace lowest confidence point
                auto min_confidence_it = std::min_element(
                    voxel_points.begin(), voxel_points.end(),
                    [](const auto &a, const auto &b) {
                        return a(7) < b(7);  // Compare confidence values
                    });

                // Only replace if new point has higher confidence
                if (new_confidence > (*min_confidence_it)(7)) {
                    *min_confidence_it = point;
                }
            } else {
                voxel_points.emplace_back(point);
            }
        } else {
            std::vector<PointWithNormal> voxel_points;
            voxel_points.reserve(max_points_per_voxel_);
            voxel_points.emplace_back(point);
            map_.insert({voxel, std::move(voxel_points)});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;

    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;
        const auto &pt = voxel_points.front();
        if ((pt.template head<3>() - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace kiss_icp
