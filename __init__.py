"""
@author: l-comm
@title: Watermark Removal
@nickname: Watermark Removal
@description: Remove watermark
"""

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "FindWatermarkNode": FindWatermarkNode,
    "RemoveWatermarkNode": RemoveWatermarkNode
}