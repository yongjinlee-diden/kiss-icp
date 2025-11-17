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
                       // Transform normal (nx, ny, nz) - rotation only
                       transformed.template segment<3>(3) = R * point.template segment<3>(3);
                       // Preserve consistency
                       transformed(6) = point(6);  // consistency
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
            double new_consistency = point(6);  // Extract consistency from new point

            if (voxel_points.size() >= max_points_per_voxel_) {
                // Consistency-based quality control: replace lowest consistency point
                auto min_consistency_it = std::min_element(
                    voxel_points.begin(), voxel_points.end(),
                    [](const auto &a, const auto &b) {
                        return a(6) < b(6);  // Compare consistency values
                    });

                // Only replace if new point has higher consistency
                if (new_consistency > (*min_consistency_it)(6)) {
                    *min_consistency_it = point;
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
        voxel_confidence_[voxel] = 1.0;  // Initialize with full confidence
        voxel_last_update_time_[voxel] = current_update_time_;
        voxel_fov_entry_time_[voxel] = current_update_time_;
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = max_distance_ * max_distance_;

    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;
        const auto &pt = voxel_points.front();
        if ((pt.template head<3>() - origin).squaredNorm() >= (max_distance2)) {
            it = map_.erase(it);
            // Also remove from confidence map and timestamp maps
            voxel_confidence_.erase(voxel);
            voxel_last_update_time_.erase(voxel);
            voxel_fov_entry_time_.erase(voxel);
        } else {
            ++it;
        }
    }
}

void VoxelHashMap::UpdateVoxelConfidence(
    const Sophus::SE3d &robot_pose,
    const std::vector<CameraParams> &camera_params,
    const std::vector<std::vector<float>> &depth_maps) {

    if (camera_params.size() != depth_maps.size() || camera_params.empty()) return;

    // Transform from global frame to robot frame
    Sophus::SE3d global2robot = robot_pose.inverse();

    // Iterate through all voxels in the map
    for (auto it = map_.begin(); it != map_.end();) {
        const auto &[voxel, voxel_points] = *it;

        // Skip voxels that were updated in the current step
        if (voxel_last_update_time_[voxel] >= current_update_time_) {
            ++it;
            continue;
        }

        // Use first point in voxel as representative
        const auto &pt = voxel_points.front();
        Eigen::Vector3d pt_global = pt.template head<3>();

        // Transform point from global to robot frame
        Eigen::Vector3d pt_robot = global2robot * pt_global;

        // Check visibility and occlusion against all cameras
        bool was_observed = false;
        bool is_occluded = false;

        for (size_t cam_idx = 0; cam_idx < camera_params.size(); ++cam_idx) {
            const auto &cam = camera_params[cam_idx];
            const auto &depth_map = depth_maps[cam_idx];

            if (depth_map.size() != static_cast<size_t>(cam.width * cam.height)) {
                continue;  // Skip invalid depth map
            }

            // Project point from robot frame to image plane
            Eigen::Vector4d pt_robot_h(pt_robot.x(), pt_robot.y(), pt_robot.z(), 1.0);
            Eigen::Vector4d pt_img_h = cam.robot2img_transform * pt_robot_h;

            // Check if point is behind camera
            if (pt_img_h(2) <= 0.01) continue;

            // Perspective division to get pixel coordinates
            double u_f = pt_img_h(0) / pt_img_h(2);
            double v_f = pt_img_h(1) / pt_img_h(2);
            int u = static_cast<int>(std::round(u_f));
            int v = static_cast<int>(std::round(v_f));

            // Check if pixel is within image bounds
            if (u < 0 || u >= cam.width || v < 0 || v >= cam.height) continue;
            voxel_fov_entry_time_[voxel] = current_update_time_;

            // Get depth from depth map
            int pixel_idx = v * cam.width + u;
            float observed_depth = depth_map[pixel_idx];

            // Skip invalid depth
            if (observed_depth <= 0.0f) continue;

            was_observed = true;

            // Get voxel depth (distance along camera z-axis)
            double voxel_depth = pt_img_h(2);

            // Check if voxel is occluded (behind observed surface)
            if (voxel_depth < (observed_depth - depth_tolerance_)) {
                is_occluded = true;
                break;  // No need to check other cameras
            }
        }

        // Update confidence based on observation
        if (was_observed) {
            if (is_occluded) {
                // Voxel is occluded (likely dynamic object) - decrease confidence
                voxel_confidence_[voxel] -= decay_rate_;
            }

            // Remove voxel if confidence is too low
            if (voxel_confidence_[voxel] < confidence_threshold_) {
                it = map_.erase(it);
                voxel_confidence_.erase(voxel);
                voxel_last_update_time_.erase(voxel);
                voxel_fov_entry_time_.erase(voxel);
                continue;
            }
        }

        ++it;
    }
}

}  // namespace kiss_icp
