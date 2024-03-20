import numpy as np
def relative_coordinate_y(f:float,B: float,v:list[float],h:float):
    '''
    f: focal length of camera
    B: gimbal angle, same as gimbal pitch
    camera roll angle assumed to be 0 as we're using gimbal
    v: y axis pixel position offset from the horizontal center pixel line of the image
    h: vertical ground distance of the drone
    '''
    a = np.arctan(v/f)
    y = np.tan(90-(B-a))*h
    return y

def relative_coordinate_x(h:float,y:float,f:float,v:list[float], u:list[float]):
    '''
    h: vertical ground distance of the drone
    y: relative y coordinate
    f: focal length of the camera
    v: y axis pixel position offset from the horizontal center pixel line of the image
    u: x axis pixel offset from the vertical center pixel line of the image
    '''
    d = np.sqrt(np.square(h) + np.square(y))
    w = np.sqrt(np.square(f) + np.square(v))
    x = (np.divide(u,w).T)*d
    return x

def relative_coordinates(x_center: float, y_center:float, f:float,h:float,B:float, x_det: list[float], y_det:list[float]):
    '''
    x_center, y_center: x,y coordinate of the center of the image
    f: focal length of the camera
    h: vertical ground distance of the drone
    B: gimbal angle, same as gimbal pitch
    det: x,y of the center of detection [[x1,y1],[x2,y2],...] where x1,y1 are detection 1 x2,y2 is detection 2 and so on
    '''

    u = np.array(x_det)-x_center
    v = np.array(y_det)-y_center
    y = relative_coordinate_y(f,B,v,h)
    x = relative_coordinate_x(h,y,f,v,u)
    return x,y

def estimate_horizon(h:float, f:float, B: float):
    '''
    h: vertical ground distance of the drone
    f: focal length
    B: gimbal pitch
    -------------
    RETURNS:
    l: distance to the horizon line
    gamma: angle to the horizon line
    o: offset of horizon line from the horizontal center line
    '''
    l= 3.57*(h**0.5)
    gamma = np.arcsin(h/l)
    o = np.tan(abs(gamma - B))*f*np.sign(gamma-B)

    return l,gamma,o

def radius_earth(la: float):
    '''
    la: latitude of the camera
    '''
    e = 0.0818 #eccentricity of earth ellipsoid
    return 6378*np.sqrt((1-(2*(e**2)-(e**4))*np.sin(np.deg2rad(la))**2)/(1-(e**2 * np.sin(np.deg2rad(la))**2)))

def absolute_coordinates(img_shape,r: float,h:float,B:float, x_det: list[float],y_det: list[float], la:float, lo: float,f:float = 12.5):
    '''
    img_shape: image.shape (y,x,3)
    h: vertical ground distance of the drone
    B: gimbal pitch
    f: focal length
    r: camera heading angle
    la: latitude of drone
    lo: longitude of drone
    -------------------------
    RETURNS
    la_obj: latitude of object
    lo_obj: longitude of object
    '''
    B = np.deg2rad(B)
    r = np.deg2rad(r)
    h = h/1000 # meters  to kilometers
    R = radius_earth(la)
    x_center = img_shape[1]/2
    y_center = img_shape[0]/2
    x,y = relative_coordinates(x_center,y_center,f,h,B,x_det,y_det)
    x_r = x*np.cos(r) + y*np.sin(r)
    y_r = x*np.sin(r) + y*np.cos(r)
    la_obj = la + np.rad2deg(y_r/R)
    lo_obj = lo + np.rad2deg(x_r/R*(1/np.cos(np.deg2rad(la))))    #cHECK ONCE MIGHT BE WRONG  
    return [(i,j) for i,j in zip(la_obj, lo_obj)]


# import json
# import pandas as pd

# with open('annotations/instances_train_objects_in_water.json') as f:
#     data = json.load(f)
# print(data.keys())
# df = pd.DataFrame.from_dict(data['images'])
# print(df['meta'][0])
# print(np.arctan(1))
# print(np.tan(90 - np.arctan(1)))
# # # gps_latitude': 47.671755, 'gps_latitude_ref': 'N', 
# 'gps_longitude': 9.269907, 'gps_longitude_ref': 'E', 
# # 'altitude': 11.299448948491314, 'gimbal_pitch': 45.4, 
# 'compass_heading': 319.3, 'gimbal_heading': 322.4, 
# 'speed': 2.3429371569065713, 'xspeed': 1.7999517210549845, 
# 'yspeed': -1.4999597675458203, 'zspeed': 0.0