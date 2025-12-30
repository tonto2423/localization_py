from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def create_line_marker(
        node_clock, 
        marker_id: int, 
        namespace: str, 
        frame_id: str, 
        start_point: list[float], 
        end_point: list[float], 
        color, 
        thickness=0.025
    ):
    """2点間に線を引くためのMarkerメッセージを作成するヘルパー関数"""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = node_clock.now().to_msg()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = thickness # Line width
    marker.color = color

    p1 = Point(x=float(start_point[0]), y=float(start_point[1])) # 2D(x,z) -> 3D(x,y,z)
    p2 = Point(x=float(end_point[0]), y=float(end_point[1]))   # 2D(x,z) -> 3D(x,y,z)

    marker.points.append(p1)
    marker.points.append(p2)
    return marker