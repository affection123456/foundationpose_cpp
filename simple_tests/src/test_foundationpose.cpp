#include "tests/fps_counter.h"
#include "tests/help_func.hpp"
#include "tests/fs_util.hpp"

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>

using namespace inference_core;
using namespace detection_6d;
using json = nlohmann::json;

static const std::string refiner_engine_path_ = "../models/refiner_hwc_dynamic_fp16.engine";
static const std::string scorer_engine_path_  = "../models/scorer_hwc_dynamic_fp16.engine";
static const std::string demo_data_path_      = "../test_data/mustard0";
static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
static const std::string demo_name_             = "mustard";
static const std::string frame_id               = "1581120424100262102";
// Switch primary output frame. true: bbox frame, false: mesh frame.
static const bool        use_bbox_frame_output_  = true;
// Switch visualization frame. true: OBB/bbox frame, false: mesh-AABB frame.
static const bool        use_bbox_frame_visualization_ = true;


// static const std::string refiner_engine_path_ = "../models/refiner_hwc_dynamic_fp16.engine";
// static const std::string scorer_engine_path_  = "../models/scorer_hwc_dynamic_fp16.engine";
// static const std::string demo_data_path_      = "../test_data/magazine";
// static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
// static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
// static const std::string demo_name_             = "magazine";
// static const std::string frame_id               = "000000";
// Switch primary output frame. true: bbox frame, false: mesh frame.
// static const bool        use_bbox_frame_output_  = false;
// // Switch visualization frame. true: OBB/bbox frame, false: mesh-AABB frame.
// static const bool        use_bbox_frame_visualization_ = false;


// Align with FoundationPose_torch defaults for fair comparison.
static const size_t      register_refine_itr_   = 5;
static const size_t      track_refine_itr_      = 2;


std::tuple<std::shared_ptr<Base6DofDetectionModel>, std::shared_ptr<BaseMeshLoader>> CreateModel()
{
  auto refiner_core = CreateTrtInferCore(refiner_engine_path_,
                                         {
                                             {"transf_input", {252, 160, 160, 6}},
                                             {"render_input", {252, 160, 160, 6}},
                                         },
                                         {{"trans", {252, 3}}, {"rot", {252, 3}}}, 1);
  auto scorer_core  = CreateTrtInferCore(scorer_engine_path_,
                                         {
                                            {"transf_input", {252, 160, 160, 6}},
                                            {"render_input", {252, 160, 160, 6}},
                                        },
                                         {{"scores", {252, 1}}}, 1);

  Eigen::Matrix3f intrinsic_in_mat = ReadCamK(demo_data_path_ + "/cam_K.txt");

  auto mesh_loader = CreateAssimpMeshLoader(demo_name_, demo_textured_obj_path);
  CHECK(mesh_loader != nullptr);

  auto foundation_pose =
      CreateFoundationPoseModel(refiner_core, scorer_core, {mesh_loader}, intrinsic_in_mat);

  return {foundation_pose, mesh_loader};
}

TEST(foundationpose_test, test)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  const Eigen::Vector3f object_dimension =
      use_bbox_frame_visualization_ ? mesh_loader->GetObjectDimension()
                                    : ComputeMeshAabbDimension(mesh_loader);

  Eigen::Matrix4f out_pose_mesh;
  CHECK(foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose_mesh,
                                  register_refine_itr_));
  Eigen::Matrix4f out_pose = ConvertPoseForOutput(out_pose_mesh, mesh_loader, use_bbox_frame_output_);
  LOG(WARNING) << "first Pose : " << out_pose;

  // JSON object to store all frame poses
  json poses_json;
  // Debug-only: store pose converted to bbox frame for coordinate-system comparison.
  json poses_bbox_json;

  // Save first frame pose
  auto [trans, quat] = PoseToTransQuat(out_pose);
  poses_json[frame_id] = {trans.x(), trans.y(), trans.z(), 
                          quat.x(), quat.y(), quat.z(), quat.w()};
  auto [bbox_trans, bbox_quat] = PoseToTransQuat(ConvertPoseMesh2BBox(out_pose_mesh, mesh_loader));
  poses_bbox_json[frame_id] = {bbox_trans.x(), bbox_trans.y(), bbox_trans.z(), bbox_quat.x(),
                               bbox_quat.y(), bbox_quat.z(), bbox_quat.w()};
  auto [mesh_trans, mesh_quat] = PoseToTransQuat(out_pose_mesh);
  LogQuatCompare(frame_id, mesh_quat, bbox_quat);

  // [temp] for test
  cv::Mat regist_plot = rgb.clone();
  cv::cvtColor(regist_plot, regist_plot, cv::COLOR_RGB2BGR);
  auto draw_pose = use_bbox_frame_visualization_
                       ? ConvertPoseMesh2BBox(out_pose_mesh, mesh_loader)
                       : ConvertPoseMesh2AabbCenter(out_pose_mesh, mesh_loader);
  draw3DBoundingBox(intrinsic_in_mat, draw_pose, 480, 640, object_dimension, regist_plot);
  cv::imwrite(demo_data_path_ + "/test_foundationpose_plot.png", regist_plot);

  auto rgb_paths = get_files_in_directory(demo_data_path_ + "/rgb/");
  std::sort(rgb_paths.begin(), rgb_paths.end());
  std::vector<std::string> frame_ids;
  for (const auto &rgb_path : rgb_paths)
  {
    frame_ids.push_back(rgb_path.stem());
  }

  int                  total = frame_ids.size();
  std::vector<cv::Mat> result_image_sequence{regist_plot};
  for (int i = 1; i < total; ++i)
  {
    std::string cur_rgb_path   = demo_data_path_ + "/rgb/" + frame_ids[i] + ".png";
    std::string cur_depth_path = demo_data_path_ + "/depth/" + frame_ids[i] + ".png";
    auto [cur_rgb, cur_depth]  = ReadRgbDepth(cur_rgb_path, cur_depth_path);

    Eigen::Matrix4f track_pose_mesh;
    CHECK(foundation_pose->Track(cur_rgb.clone(), cur_depth, out_pose_mesh, demo_name_, track_pose_mesh,
                                 track_refine_itr_));
    Eigen::Matrix4f track_pose =
        ConvertPoseForOutput(track_pose_mesh, mesh_loader, use_bbox_frame_output_);
    LOG(WARNING) << "Track pose : " << track_pose;

    // Save current frame pose to JSON
    auto [cur_trans, cur_quat] = PoseToTransQuat(track_pose);
    poses_json[frame_ids[i]] = {cur_trans.x(), cur_trans.y(), cur_trans.z(),
                                cur_quat.x(), cur_quat.y(), cur_quat.z(), cur_quat.w()};
    auto [cur_bbox_trans, cur_bbox_quat] =
        PoseToTransQuat(ConvertPoseMesh2BBox(track_pose_mesh, mesh_loader));
    poses_bbox_json[frame_ids[i]] = {cur_bbox_trans.x(), cur_bbox_trans.y(), cur_bbox_trans.z(),
                                     cur_bbox_quat.x(), cur_bbox_quat.y(), cur_bbox_quat.z(),
                                     cur_bbox_quat.w()};
    auto [cur_mesh_trans, cur_mesh_quat] = PoseToTransQuat(track_pose_mesh);
    LogQuatCompare(frame_ids[i], cur_mesh_quat, cur_bbox_quat);

    cv::Mat track_plot = cur_rgb.clone();
    cv::cvtColor(track_plot, track_plot, cv::COLOR_RGB2BGR);
    auto draw_pose = use_bbox_frame_visualization_
                         ? ConvertPoseMesh2BBox(track_pose_mesh, mesh_loader)
                         : ConvertPoseMesh2AabbCenter(track_pose_mesh, mesh_loader);
    draw3DBoundingBox(intrinsic_in_mat, draw_pose, 480, 640, object_dimension, track_plot);
    cv::imshow("test_foundationpose_result", track_plot);
    cv::waitKey(20);
    result_image_sequence.push_back(track_plot);

    out_pose_mesh = track_pose_mesh;
    out_pose      = track_pose;
  }

  // Save JSON to file
  std::string json_output_path = demo_data_path_ + "/poses_result.json";
  std::ofstream json_file(json_output_path);
  json_file << poses_json.dump(2) << std::endl;
  json_file.close();
  LOG(WARNING) << "Poses saved to: " << json_output_path;

  std::string bbox_json_output_path = demo_data_path_ + "/poses_result_bbox.json";
  std::ofstream bbox_json_file(bbox_json_output_path);
  bbox_json_file << poses_bbox_json.dump(2) << std::endl;
  bbox_json_file.close();
  LOG(WARNING) << "BBox-frame poses saved to: " << bbox_json_output_path;

  saveVideo(result_image_sequence, demo_data_path_ + "/test_foundationpose_result.mp4");
}

TEST(foundationpose_test, speed_register)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0; i < 50; ++i)
  {
    Eigen::Matrix4f out_pose;
    foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();
}

TEST(foundationpose_test, speed_track)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  Eigen::Matrix4f first_pose;
  foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, first_pose);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0; i < 5000; ++i)
  {
    Eigen::Matrix4f track_pose;
    foundation_pose->Track(rgb.clone(), depth, first_pose, demo_name_, track_pose);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();
}
