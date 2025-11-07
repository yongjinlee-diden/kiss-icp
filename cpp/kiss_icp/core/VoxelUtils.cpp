#include "VoxelUtils.hpp"

#include <tsl/robin_map.h>

namespace kiss_icp {

std::vector<Eigen::Vector4d> VoxelDownsample(const std::vector<Eigen::Vector4d> &frame,
                                             const double voxel_size) {
    tsl::robin_map<Voxel, Eigen::Vector4d> grid;
    grid.reserve(frame.size());
    std::for_each(frame.cbegin(), frame.cend(), [&](const auto &point) {
        // Extract x, y, z for voxel computation
        Eigen::Vector3d point_xyz = point.template head<3>();
        const auto voxel = PointToVoxel(point_xyz, voxel_size);
        if (!grid.contains(voxel)) grid.insert({voxel, point});
    });
    std::vector<Eigen::Vector4d> frame_dowsampled;
    frame_dowsampled.reserve(grid.size());
    std::for_each(grid.cbegin(), grid.cend(), [&](const auto &voxel_and_point) {
        frame_dowsampled.emplace_back(voxel_and_point.second);
    });
    return frame_dowsampled;
}

}  // namespace kiss_icp
