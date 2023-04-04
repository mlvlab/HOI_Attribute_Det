from torch import nn
from torchvision.ops import roi_align


# NOTE: torchvision's RoIAlign has a different default aligned=False
class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.
        Note:
            The meaning of aligned=True:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.
            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.
            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """

        '''
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN
        
        Parameters
        input (Tensor[N, C, H, W]) – input tensor
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from. The coordinate must satisfy 0 <= x1 < x2 and 0 <= y1 < y2. If a single Tensor is passed, then the first column should contain the batch index. If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i in a batch
        output_size (int or Tuple[int, int]) – the size of the output after the cropping is performed, as (height, width)
        spatial_scale (float) – a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0
        sampling_ratio (int) – number of sampling points in the interpolation grid used to compute the output value of each pooled output bin. If > 0, then exactly sampling_ratio x sampling_ratio grid points are used. If <= 0, then an adaptive number of grid points are used (computed as ceil(roi_width / pooled_w), and likewise for height). Default: -1
        aligned (bool) – If False, use the legacy implementation. If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices. This version in Detectron2
        Returns
        output (Tensor[K, C, output_size[0], output_size[1]])
        '''
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

        from torchvision import __version__

        version = tuple(int(x) for x in __version__.split(".")[:2])
        # https://github.com/pytorch/vision/pull/2438
        assert version >= (0, 7), "Require torchvision >= 0.7"

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        if input.is_quantized:
            input = input.dequantize()
        return roi_align(
            input,
            rois.to(dtype=input.dtype),
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.aligned,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr