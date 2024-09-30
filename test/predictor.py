from typing import IO
import onnxruntime as ort
import numpy as np
import torch
import yaml
from train.config.model_config_IA import IA_segmentation_cfg
from train.config.model_config_stenosis import stenosis_segmentation_cfg
import tensorrt as trt
import pycuda.driver as pdd
import pycuda.autoinit
import SimpleITK as sitk
import itk
from skimage.measure import label
from scipy.ndimage import filters
from skimage.transform import resize

def get_auto_windowing(image, lower_percentile=0.1, upper_percentile=99.9):
    
    """
    计算图像直方图的低和高百分位数。
    """
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 256))
    
    # 计算累计直方图
    cumulative_hist = np.cumsum(hist)
    cumulative_hist = cumulative_hist / cumulative_hist[-1] 
    
    # 计算低和高百分位数
    lower_threshold = np.percentile(image, lower_percentile)
    upper_threshold = np.percentile(image, upper_percentile)

    return lower_threshold, upper_threshold


def resample_image(image, new_spacing):
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    # 获取原始 spacing 和尺寸
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 计算新的尺寸
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(len(new_spacing))]

    # 创建 resample 参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear) 

    # 执行 resample 操作
    resampled_image = resampler.Execute(image)

    return resampled_image

def get_outer_box(mask_array):
    foreground_indices = np.argwhere(mask_array == 1)
    if foreground_indices.size == 0:
        raise ValueError("No foreground pixels found in the mask.")

    min_coords = foreground_indices.min(axis=0)
    max_coords = foreground_indices.max(axis=0) + 1 
    return [min_coords, max_coords]


def crop_and_pad(img, box_coords, target_size):
    min_coords, max_coords = box_coords
    # 裁剪图像
    cropped_img = img[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
        ]

    # 填充图像至固定尺度
    pad_size = [max(0, target_size[i] - cropped_img.shape[i]) for i in range(3)]
    pad_left = [pad_size[i] // 2 for i in range(3)]
    pad_right = [pad_size[i] - pad_left[i] for i in range(3)]
    padded_img = np.pad(cropped_img, 
                         pad_width=[(pad_left[0], pad_right[0]), 
                                    (pad_left[1], pad_right[1]), 
                                    (pad_left[2], pad_right[2])], 
                         mode='constant', constant_values=0)

    return padded_img


def get_crop_area(mask, crop_size):
    # 获取外接立方体的边界框
    z_min, y_min, x_min = np.min(np.nonzero(mask), axis=1)
    z_max, y_max, x_max = np.max(np.nonzero(mask), axis=1)

    # 计算外接立方体的中心
    center_of_mass = np.array([(z_min + z_max) // 2,
                               (y_min + y_max) // 2,
                               (x_min + x_max) // 2])
    
    # 获取裁剪区域的尺寸
    depth, height, width = crop_size

    # 计算裁剪区域的边界
    start_z = max(z_max - depth, 0)
    start_y = max(center_of_mass[1] - height // 2, 0)
    start_x = max(center_of_mass[2] - width // 2, 0)
    
    end_z = min(start_z + depth, mask.shape[0])
    end_y = min(start_y + height, mask.shape[1])
    end_x = min(start_x + width, mask.shape[2])

    # 调整起始位置以适应边界条件
    start_y = end_y - depth if end_y == mask.shape[1] else start_y
    start_x = end_x - width if end_x == mask.shape[2] else start_x

    crop_area = np.zeros_like(mask, dtype=np.uint8)
    crop_area[start_z:end_z, start_y:end_y, start_x:end_x] = 1

    return crop_area


def get_longest_components(mask, num_components):
    # 使用连通域分析找出最长的连通区域
    labeled_mask = label(mask, connectivity=3) 
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # 背景的体积不计入
    component_volumes = [(i, sizes[i]) for i in range(1, len(sizes)) if sizes[i] > 0]
    # 按体积排序并选取前 num_components 个
    longest_components = sorted(component_volumes, key=lambda x: x[1], reverse=True)[:num_components]
    # print(longest_components)
    largest_component_indices = {i[0] for i in longest_components if i[1] > 6000}
    
    # 只保留这些最大的连通区域
    filtered_mask = np.isin(labeled_mask, list(largest_component_indices)).astype(np.uint8)
    return filtered_mask
    

def vesselSegment(input_image):

    Dimension = input_image.GetImageDimension()
    ImageType = type(input_image)

    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(0.5)
    objectness_filter.SetGamma(2.0)
    
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(0.1)
    multi_scale_filter.SetSigmaMaximum(5.0)
    multi_scale_filter.SetNumberOfSigmaSteps(6)
    
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    
    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter.GetOutput())

    enhanced_img = itk.GetArrayFromImage(rescale_filter.GetOutput())

    crop_area = get_crop_area(enhanced_img > 80, [256, 256, 256])

    vessel_mask = (enhanced_img > 20) * crop_area
    vessel_mask = get_longest_components(vessel_mask, 6)

    return vessel_mask.astype("uint8")


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class PredictConfig:
    def __init__(self, test_cfg):
        # 配置文件
        self.IA_img_size = test_cfg["IA"]['img_size']
        self.stenosis_img_size = test_cfg["stenosis"]['img_size']
        self.IA_seg_thresholds = test_cfg["IA"]['seg_thresholds']
        self.stenosis_seg_thresholds = test_cfg["stenosis"]['seg_thresholds']

    def __repr__(self) -> str:
        return str(self.__dict__)


class PredictModel:
    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f


class Predictor:
    def __init__(self, device: str, task: str, model: PredictModel):
        self.device = torch.device(device)
        self.task = task
        self.model = model
        self.dym_flag = False
        self.jit_flag = False
        self.tensorrt_flag = False 
        self.ort_flag = False

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = PredictConfig(yaml.safe_load(config_f))

        if task == "stenosis":
            self.network_cfg = stenosis_segmentation_cfg
        elif task == "IA":
            self.network_cfg = IA_segmentation_cfg
        else:
            raise ValueError("Task must be stenosis or IA")
        
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.dym_flag = True
                self.load_model_pth()
            elif self.model.model_f.endswith('.pt'):
                self.jit_flag = True
                self.load_model_jit()
            elif self.model.model_f.endswith('.onnx'):
                self.ort_flag = True
                self.load_model_onnx()
            elif self.model.model_f.endswith('.engine'):
                self.tensorrt_flag = True
                self.load_model_engine()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device)

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def load_model_onnx(self) -> None:
        # 加载onnx
        self.ort_session = ort.InferenceSession(self.model.model_f, providers=['CUDAExecutionProvider'])

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model.model_f, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = pdd.Stream()
        for i, binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = pdd.pagelocked_empty(size, dtype)
            device_mem = pdd.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


    def trt_inference(self, context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [pdd.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [pdd.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def torch_predict(self, img):
        with torch.no_grad():
            pred = self.net(img.to(self.device))
        return pred

    def ort_predict(self, img):
        pred = self.ort_session.run(None, {"input": img.numpy()})     
        return pred

    def trt_predict(self, img):
        cuda_ctx = pycuda.autoinit.context
        cuda_ctx.push()
        img = np.ascontiguousarray(img.numpy().astype(np.float32))
        self.context.active_optimization_profile = 0
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3] = img.shape
        # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size
        self.context.set_binding_shape(0, (origin_inputshape)) 
        inputs, outputs, bindings, stream = self.allocate_buffers(self.engine, self.context)
        inputs[0].host = img
        trt_outputs = self.trt_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,stream=stream, batch_size=1)
        if cuda_ctx:
            cuda_ctx.pop()
        pred = trt_outputs[0]
        return pred
    
    def pre_process(self, sitk_img):
        if self.task == "IA":
            self.img_size = self.test_cfg.IA_img_size
            gaussian_sigma = 3
        else:
            self.img_size = self.test_cfg.stenosis_img_size
            gaussian_sigma = 0.5

        self.ori_size  = sitk.GetArrayFromImage(sitk_img).shape

        sitk_img = resample_image(sitk_img, [0.45, 0.45, 0.45])
        img = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        self.resampled_size = img.shape

        itk_img = itk.GetImageFromArray(img)
        vesselmask = vesselSegment(itk_img)

        self.crop_coords = get_outer_box(vesselmask)
        img_patch = crop_and_pad(img, self.crop_coords, self.img_size)
        vesselmask_patch = crop_and_pad(vesselmask, self.crop_coords, self.img_size)
        vessel_hint_patch = filters.gaussian_filter(vesselmask_patch.astype("float"), gaussian_sigma)
        self.vessel_mask = vessel_hint_patch > 0

        min_val, max_val = get_auto_windowing(img_patch, lower_percentile=0.1, upper_percentile=99.9)
        img_patch = np.clip(img_patch, min_val, max_val)
        img_patch = (img_patch - min_val)/(max_val - min_val)

        img_t = torch.from_numpy(img_patch)[None, None]
        vessel_hint_t = torch.from_numpy(vessel_hint_patch)[None, None]

        return torch.cat([img_t.float(), vessel_hint_t.float()], 1)

    def post_porcess(self, pred):
        if self.task == "IA":
            seg_thresholds = self.test_cfg.IA_seg_thresholds
        else:
            seg_thresholds = self.test_cfg.stenosis_seg_thresholds

        
        pred_array = pred.cpu().squeeze().numpy().astype("float32")
        pred_array = self.vessel_mask * pred_array

         # 计算裁剪区域和填充区域
        min_coords, max_coords = self.crop_coords
        cropped_size = [max_coords[i] - min_coords[i] for i in range(3)]
        pad_size = [max(0, self.img_size[i] - cropped_size[i]) for i in range(3)]
        pad_left = [pad_size[i] // 2 for i in range(3)]
        pad_right = [pad_size[i] - pad_left[i] for i in range(3)]
        crop_slices = tuple(slice(pad_left[i], pred_array.shape[i] - pad_right[i]) for i in range(3))
        cropped_pred = pred_array[crop_slices]
        restored_array = np.zeros(self.resampled_size, dtype=pred_array.dtype)
        restore_slices = tuple(slice(min_coords[i], min_coords[i] + cropped_pred.shape[i]) for i in range(3))
        restored_array[restore_slices] = cropped_pred        
        heatmap = resize(restored_array, self.ori_size, mode='constant', anti_aliasing=True, order=3)
        seg = (heatmap > seg_thresholds).astype("uint8")
        return seg, heatmap

    def predict(self, sitk_img):

        img_t = self.pre_process(sitk_img)

        if self.dym_flag or self.jit_flag:
            pred =  self.torch_predict(img_t)
        elif self.ort_flag:
            pred =  self.ort_predict(img_t)
        elif self.tensorrt_flag:
            pred = self.trt_predict(img_t)
        else:
            raise NotImplementedError

        seg, heatmap = self.post_porcess(pred)

        return seg, heatmap





