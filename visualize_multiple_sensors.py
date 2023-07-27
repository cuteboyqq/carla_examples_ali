#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""
#from carla import ColorConverter as cc
import glob
import os
import sys
import numpy
from multiprocessing import Process, Lock
mutex = Lock()
i=1
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)
        self.recording = False
        self.period = 40
    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar
        
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        
        elif sensor_type == 'semantic_segmentation':

            sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            disp_size = self.display_man.get_display_size()
            sem_bp.set_attribute('image_size_x', str(disp_size[0]))
            sem_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                sem_bp.set_attribute(key, sensor_options[key])

            sem_cam = self.world.spawn_actor(sem_bp, transform, attach_to=attached)

            sem_cam.listen(self.save_semantic_img)
            #sem_cam.listen(lambda image: image.save_to_disk('ali_sem2/%.6d.jpg' % image.frame))

            return sem_cam
        elif sensor_type == 'instance_segmentation':

            ins_sem_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
            disp_size = self.display_man.get_display_size()
            ins_sem_bp.set_attribute('image_size_x', str(disp_size[0]))
            ins_sem_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                ins_sem_bp.set_attribute(key, sensor_options[key])

            ins_sem_cam = self.world.spawn_actor(ins_sem_bp, transform, attach_to=attached)

            ins_sem_cam.listen(self.save_instance_semantic_img)
            #sem_cam.listen(lambda image: image.save_to_disk('ali_sem2/%.6d.jpg' % image.frame))

            return ins_sem_cam
        elif sensor_type == 'depthCamera':
            depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            disp_size = self.display_man.get_display_size()
            depth_bp.set_attribute('image_size_x', str(disp_size[0]))
            depth_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                depth_bp.set_attribute(key, sensor_options[key])

            depth_cam = self.world.spawn_actor(depth_bp, transform, attach_to=attached)

            depth_cam.listen(self.save_depth_img)

            return depth_cam
        else:
            return None

    def get_sensor(self):
        return self.sensor

    #===============================================================================================================================
    def to_bgra_array(self,image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.width, image.height, 4))
        return array

    def to_rgb_array(self,image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self._camera_transformso_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array
    
    def labels_to_array(self,image):
        """
        Convert an image containing CARLA semantic segmentation labels to a 2D array
        containing the label of each pixel.
        """
        return self.to_bgra_array(image)[:, :, 2]
        #return self.to_rgb_array(image)

    def labels_to_cityscapes_palette(self,image):
        """
        Convert an image containing CARLA semantic segmentation labels to
        Cityscapes palette.
        """
        classes = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]     # TrafficSigns
        }
        array = self.labels_to_array(image)
        print(array)
        result = numpy.zeros((array.shape[0], array.shape[1], 3))
        for key, value in classes.items():
            mutex.acquire()
            print(key, numpy.where(array == key))
            mutex.release()
            result[numpy.where(array == key)] = value
        return result

    def save_semantic_img(self, image):
        global i
        image.convert(carla.ColorConverter.CityScapesPalette)  # rgb
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and image.frame % self.period == 0:
            image.save_to_disk('sem_out_ali/%08d' % image.frame)
        
        return array
    
    def save_instance_semantic_img(self,image):
        image.convert(carla.ColorConverter.Raw)  # rgb
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and image.frame % self.period == 0:
            image.save_to_disk('sem_out_ali/%08d' % image.frame)
        
        return array
    
    def save_depth_img(self, image):
        global i
        image.convert(carla.ColorConverter.LogarithmicDepth)  # rgb
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and image.frame % self.period == 0:
            image.save_to_disk('depth_out_ali/%08d' % image.frame)
        
        return array
    #===============================================================================================================================
    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

        if self.recording and image.frame % self.period == 0:
            image.save_to_disk('ori_out_ali/%08d' % image.frame)

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)


        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width * 3, args.height * 2])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1.0, z=1.2), carla.Rotation(yaw=0)), 
                      vehicle, {}, display_pos=[0, 0])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), 
        #               vehicle, {}, display_pos=[0, 1])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)), 
        #               vehicle, {}, display_pos=[0, 2])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), 
        #               vehicle, {}, display_pos=[1, 1])

        SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=1.2)), 
                       vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=[0, 2])
        SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0, z=1.2)), 
                       vehicle, {'channels' : '64', 'range' : '100', 'points_per_second': '100000', 'rotation_frequency': '20'}, display_pos=[1, 0])
        
        SensorManager(world, display_manager, 'semantic_segmentation', carla.Transform(carla.Location(x=1.0, z=1.2), carla.Rotation(yaw=0)), 
                      vehicle, {}, display_pos=[0, 1])
        
        SensorManager(world, display_manager, 'depthCamera', carla.Transform(carla.Location(x=1.0, z=1.2), carla.Rotation(yaw=0)), 
                      vehicle, {}, display_pos=[1, 1])
        
        SensorManager(world, display_manager, 'instance_segmentation', carla.Transform(carla.Location(x=1.0, z=1.2), carla.Rotation(yaw=0)), 
                      vehicle, {}, display_pos=[1, 2])
        
        # SensorManager(world, display_manager, 'semantic_segmentation', carla.Transform(carla.Location(x=0, z=2.4)), 
        #               vehicle, {}, display_pos=[2, 1])
        
        # SensorManager(world, display_manager, 'semantic_segmentation', carla.Transform(carla.Location(x=0, z=2.4)), 
        #               vehicle, {}, display_pos=[2, 2])


        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)



def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    # argparser.add_argument(
    #     '--save_img',
    #     default=False,
    #     help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x384',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
