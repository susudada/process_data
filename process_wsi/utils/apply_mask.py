import openslide
import geojson
from shapely.geometry import shape

def apply_mask(slide_path, geojson_path):
    """
    从SVS文件中提取GeoJSON中定义的多边形区域。

    :param slide_path: SVS文件的路径
    :param geojson_path: GeoJSON文件的路径
    :return: 提取的区域图像列表
    """
    slide = openslide.OpenSlide(slide_path)
    
    try:
        with open(geojson_path) as f:
            gj = geojson.load(f)
        
        regions = []
        
        for feature in gj['features']:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':  # 只处理多边形
                polygon = shape(geometry)  # 将GeoJSON几何转换为Shapely多边形
                minx, miny, maxx, maxy = map(int, polygon.bounds)  # 获取并转换边界框为整数
                
                # 计算区域的宽度和高度
                width, height = maxx - minx, maxy - miny
                
                # 从SVS文件中提取区域
                level = 0  # 使用最高分辨率层级
                region = slide.read_region((minx, miny), level, (width, height))
                region = region.convert("RGB")  # 转换为RGB格式
                regions.append(region)
        
        return regions
    
    finally:
        slide.close()  # 确保幻灯片文件被关闭