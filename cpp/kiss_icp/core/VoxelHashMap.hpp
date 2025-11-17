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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "VoxelUtils.hpp"

namespace kiss_icp {

// Structure to hold point with normal vector and quality metrics:
// (x, y, z, t, nx, ny, nz, consistency)
// Using Eigen::Matrix<double, 8, 1> for better memory alignment
using PointWithNormal = Eigen::Matrix<double, 8, 1>;

struct VoxelHashMap {
    explicit VoxelHashMap(double voxel_size, double max_distance, unsigned int max_points_per_voxel,
                         double decay_rate, double depth_tolerance, double confidence_threshold)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel),
          decay_rate_(decay_rate),
          depth_tolerance_(depth_tolerance),
          confidence_threshold_(confidence_threshold) {}

    inline void Clear() {
        map_.clear();
        voxel_confidence_.clear();
        voxel_last_update_time_.clear();
        voxel_fov_entry_time_.clear();
    }
    inline bool Empty() const { return map_.empty(); }

    // Set current update time (called before Update to track when voxels are added)
    inline void SetCurrentTime(double time) { current_update_time_ = time; }

    // Unified API - always uses PointWithNormal internally
    void Update(const std::vector<PointWithNormal> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<PointWithNormal> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<PointWithNormal> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<PointWithNormal> Pointcloud() const;
    std::tuple<PointWithNormal, double> GetClosestNeighbor(const PointWithNormal &query) const;

    // Dynamic object removal via confidence-based ray casting
    // Camera parameters for projection
    struct CameraParams {
        int width, height;
        Eigen::Matrix4d robot2img_transform;  // Transform from robot frame to image plane
    };

    // Update voxel confidence based on depth observations
    // robot_pose: current robot pose (SE3 transformation from global to robot frame)
    // camera_params: camera intrinsics and extrinsics
    // depth_maps: depth images from all cameras (row-major, width x height per camera)
    // Uses stored config parameters (decay_rate_, boost_rate_, etc.)
    void UpdateVoxelConfidence(
        const Sophus::SE3d &robot_pose,
        const std::vector<CameraParams> &camera_params,
        const std::vector<std::vector<float>> &depth_maps);

    double voxel_size_;
    double max_distance_;
    unsigned int max_points_per_voxel_;

    // Dynamic removal config parameters
    double decay_rate_;
    double depth_tolerance_;
    double confidence_threshold_;

    // Single unified storage (always PointWithNormal)
    tsl::robin_map<Voxel, std::vector<PointWithNormal>> map_;

    // Per-voxel confidence for dynamic object removal
    tsl::robin_map<Voxel, double> voxel_confidence_;

    // Per-voxel timestamp tracking
    tsl::robin_map<Voxel, double> voxel_last_update_time_;  // Last time voxel was updated (added/modified)
    tsl::robin_map<Voxel, double> voxel_fov_entry_time_;    // Time when voxel first entered camera FOV

    // Current update timestamp (set before AddPoints to track update batches)
    double current_update_time_ = 0.0;
};
}  // namespace kiss_icp
