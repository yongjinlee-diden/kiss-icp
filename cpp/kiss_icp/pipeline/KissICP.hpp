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
#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tuple>
#include <vector>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/Threshold.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

struct KISSConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 0.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // registration params
    int max_num_iterations = 500;
    double convergence_criterion = 0.0001;
    int max_num_threads = 0;

    // Motion compensation
    bool deskew = true;

    // Normal vector-based ICP (inspired by nv_liom)
    bool use_normals = false;
    double normal_consistency_threshold = 0.9848;  // cos(10°)

    // Consistency-based point quality control
    bool use_consistency_weighting = true;  // Enable consistency-based weighting in ICP

    // Dynamic object removal via confidence-based ray casting
    bool dynamic_removal_enabled = false;
    double confidence_threshold = 0.8;
    double confidence_decay_rate = 0.1;
    double depth_tolerance_m = 0.05;
};

class KissICP {
public:
    using PointWithNormalVector = std::vector<PointWithNormal>;
    using PointWithNormalVectorTuple = std::tuple<PointWithNormalVector, PointWithNormalVector>;

public:
    explicit KissICP(const KISSConfig &config)
        : config_(config),
          preprocessor_(config.max_range, config.min_range, config.deskew, config.max_num_threads),
          registration_(
              config.max_num_iterations, config.convergence_criterion, config.max_num_threads,
              config.use_normals, config.normal_consistency_threshold, config.use_consistency_weighting),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel,
                     config.confidence_decay_rate, config.depth_tolerance_m,
                     config.confidence_threshold),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {}

public:
    // Unified API (automatically handles normals based on use_normals config)
    PointWithNormalVectorTuple RegisterFrame(const std::vector<PointWithNormal> &frame);
    PointWithNormalVectorTuple Voxelize(const std::vector<PointWithNormal> &frame) const;
    std::vector<PointWithNormal> LocalMap() const { return local_map_.Pointcloud(); };

    const Sophus::SE3d &pose() const { return last_pose_; }
    Sophus::SE3d &pose() { return last_pose_; }

    void Reset();

    // Dynamic object removal support
    void SetCameraParams(const std::vector<VoxelHashMap::CameraParams> &params) {
        camera_params_ = params;
    }
    void SetDepthMaps(const std::vector<std::vector<float>> &depth_maps) {
        depth_maps_ = depth_maps;
    }

    // Set current time for voxel tracking (call before RegisterFrame)
    void SetCurrentTime(double time) {
        local_map_.SetCurrentTime(time);
    }

private:
    Sophus::SE3d last_pose_;

    // KISS-ICP pipeline modules
    KISSConfig config_;
    Preprocessor preprocessor_;
    Registration registration_;
    VoxelHashMap local_map_;
    AdaptiveThreshold adaptive_threshold_;

    // Dynamic object removal
    std::vector<VoxelHashMap::CameraParams> camera_params_;
    std::vector<std::vector<float>> depth_maps_;
};

}  // namespace kiss_icp::pipeline
