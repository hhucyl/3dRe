"""One cancellable background job with its own BP mmap and immutable inputs."""

from PyQt5.QtCore import QThread, pyqtSignal

from bp_reader import BpRecording
from slice_rotation import SliceRotationResult, make_slice_corrections, optimize_recording, validate_corrections


class ReconstructionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    cloudReady = pyqtSignal(int, object, object)
    meshReady = pyqtSignal(int, object)
    spatialReady = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation, request, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.request = dict(request)

    def run(self):
        from surface_reconstruction import ReconstructionCancelled, build_surface

        request = self.request
        generation = self.generation
        cancelled = self.isInterruptionRequested
        try:
            self.progress.emit(generation, -1, "读取声呐记录")
            # Never share the GUI recording: file changes can close its mmap.
            with BpRecording(request["path"], cancelled=cancelled) as recording:
                if cancelled():
                    return
                optimization = None
                if request.get("attitude", False):
                    corrections = request.get("angles") or make_slice_corrections(recording)
                    validate_corrections(corrections)
                    boundaries = request.get("boundaries", ())
                    note = f"手动角度；结构分段 {len(boundaries)+1} 段" if boundaries else "手动角度"
                    optimization = SliceRotationResult(tuple(corrections), note, boundaries=boundaries)
                    if request.get("auto_fit", False):
                        self.progress.emit(generation, -1, "提取切片壁面，拟合 a/b/c")
                        optimization = optimize_recording(
                            recording, corrections, threshold=request["threshold"],
                            near_range=request.get("near", 0.0), max_gap=request.get("fit_gap", 0.5),
                            boundaries=boundaries,
                            cancelled=cancelled,
                            progress=lambda value, stage: self.progress.emit(generation, value, stage),
                        )
                corrections = optimization.corrections if optimization else ()
                self.progress.emit(generation, -1, "生成三维回波点云")
                cloud = recording.build_point_cloud(
                    max_points=request["max_points"], intensity_threshold=request["threshold"],
                    ring_corrections=corrections, cancelled=cancelled,
                )
                if cancelled():
                    return
                observations = None
                if request.get("spatial", False):
                    from spatial_interpolation import prepare_spatial, preview_cloud
                    observations = None if request.get("auto_fit") else request.get("cached_spatial")
                    if observations is None:
                        observations = prepare_spatial(recording, corrections, request["threshold"],
                            request.get("near", 0), spacing=request["spatial_spacing"], max_gap=request["spatial_gap"],
                            boundaries=request.get("boundaries", ()), support=request.get("support", .2),
                            denoise=request.get("denoise", True), cancelled=cancelled,
                            progress=lambda value, stage: self.progress.emit(generation, value, stage))
                    if cancelled():
                        return
                    self.spatialReady.emit(generation, observations)
                    cloud = preview_cloud(cloud, observations, request["max_points"])
                self.cloudReady.emit(generation, cloud, optimization)
                if request["surface"]:
                    mesh = request.get("cached_mesh")
                    if mesh is None:
                        mesh = build_surface(
                            recording, threshold=request["threshold"], corrections=corrections,
                            voxel_size=request["voxel"], cancelled=cancelled,
                            progress=lambda value, stage: self.progress.emit(generation, value, stage),
                            near_range=request.get("near", 0.0), connection_radius=request.get("support", 0.2),
                            denoise=request.get("denoise", True),
                            z_interpolation=request.get("z_interpolation", False),
                            z_max_gap=request.get("z_max_gap", 0.5), z_weight=request.get("z_weight", 0.3),
                            boundaries=request.get("boundaries", ()),
                            observations=observations,
                        )
                    if not cancelled():
                        self.meshReady.emit(generation, mesh)
        except ReconstructionCancelled:
            pass
        except ImportError as error:
            if not cancelled():
                self.failed.emit(generation, "缺少曲面重建依赖，请用运行程序的 Python 安装 requirements.txt。\n" + str(error))
        except Exception as error:
            if not cancelled():
                self.failed.emit(generation, str(error) or type(error).__name__)
