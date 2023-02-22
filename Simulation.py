
import time
import pandas as pd
import numpy as np
import Network as net
import Seeker
import random
from common import KM_method
import Vehicle
import os


class Simulation():

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.date = cfg.date
        self.order_list = pd.read_csv(self.cfg.order_file).sample(
            frac=cfg.demand_ratio, random_state=1)
        self.vehicle_num = int(len(self.order_list) / cfg.order_driver_ratio)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.network = net.Network()

        self.time_unit = 10  # 控制的时间窗,每10s匹配一次
        self.optimazition_target = cfg.optimazition_target  # 仿真的优化目标
        self.matching_condition = cfg.matching_condition  # 匹配时是否有条件限制
        self.pickup_distance_threshold = cfg.pickup_distance_threshold
        self.detour_distance_threshold = cfg.detour_distance_threshold
        self.vehicle_dic = {}
        self.time_reset()

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location, self.cfg)
            self.vehicle_dic[i] = vehicle
            # 重置司机的时间
            vehicle.activate_time = self.time

    def reset(self):
        self.vehicle_dic = {}
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.departure_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          >= self.begin_time]
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          <= self.end_time]

        self.time_reset()

        for i in range(self.vehicle_num):
            random.seed(i)
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location, self.cfg)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

    def time_reset(self):
        # 转换成时间数组
        self.time = time.strptime(
            self.cfg.date + self.cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        self.time = time.mktime(self.time)
        self.time_slot = 0
        # print('time reset:', self.time)

    def step(self, observation):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1
        vacant_vehicles = observation[0]
        partially_occupied_vehicles = observation[1]
        fully_occupied_vehicles = observation[2]
        passengers = observation[3]
        responsed_passengers = observation[4]

        # 实例化乘客对象
        seekers = []
        for key in passengers.keys():
            order = self.order_list[self.order_list['dwv_order_make_haikou_1.order_id'] == key]
            seeker = Seeker.Seeker(order)
            seeker.set_shortest_path(self.get_path(
                seeker.O_location, seeker.D_location))
            value = self.cfg.unit_distance_value / 1000 * seeker.shortest_distance
            seeker.set_value(value)
            seekers.append(seeker)

        # 更新司机位置
        for key in vacant_vehicles.keys():
            self.vehicle_dic[key].x = vacant_vehicles[key][0]
            self.vehicle_dic[key].y = vacant_vehicles[key][1]
            self.vehicle_dic[key].passengers = 0
            self.vehicle_dic[key].zone = self.network.getZone(self.vehicle_dic[key].x, self.vehicle_dic[key].y)

        for key in partially_occupied_vehicles.keys():
            self.vehicle_dic[key].x = partially_occupied_vehicles[key][0]
            self.vehicle_dic[key].y = partially_occupied_vehicles[key][1]
            self.vehicle_dic[key].passengers = 1
            self.vehicle_dic[key].zone = self.network.getZone(self.vehicle_dic[key].x, self.vehicle_dic[key].y)

        for key in fully_occupied_vehicles.keys():
            self.vehicle_dic[key].x = fully_occupied_vehicles[key][0]
            self.vehicle_dic[key].y = fully_occupied_vehicles[key][1]
            self.vehicle_dic[key].passengers = 2
            self.vehicle_dic[key].zone = self.network.getZone(self.vehicle_dic[key].x, self.vehicle_dic[key].y)

        start = time.time()
        done = self.process(self.time, seekers)
        end = time.time()
        # print('process 用时', end - start)
        return  done

    #
    def process(self, time_, seekers):
        takers = []
        vehicles = []

        if self.time >= time.mktime(time.strptime(self.cfg.date + self.cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S")):
            print('当前episode仿真时间结束')
            return  0, True

        else:
            for vehicle in self.vehicle_dic:
                if vehicle.passengers == 0:
                    vehicles.append(vehicle)
                elif vehicle.passengers == 1:
                    takers.append(vehicle)

            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            action = self.batch_matching(takers, vehicles, seekers)
            end = time.time()
            # print('匹配用时{},time{},vehicles{},takers{},seekers{}'.format(end - start, self.time_slot, len(vehicles), len(takers), len(seekers)))
            return  action, False

    # 匹配算法
    def batch_matching(self, takers, vehicles, seekers):
        import time
        start = time.time()
        # 构造权重矩阵
        demand = len(seekers)
        supply = len(takers) + len(vehicles)
        row_nums = demand + supply  # 加入乘客选择wait
        column_nums = demand + supply  # 加入司机选择wait
        # print('row_nums,column_nums ',row_nums,column_nums )
        dim = max(row_nums, column_nums)
        matrix = np.ones((dim, dim)) * self.cfg.dead_value

        # 从乘客角度计算匹配权重
        for column in range(demand):
            # 当前seeker的zone
            zone = self.network.getZone(seekers[column].x, seekers[column].y)
            for row in range(supply):
                if row < len(takers):
                    if takers[row].zone == zone:
                        start = time.time()
                        matrix[row, column] = self.calTakersWeights(takers[row], seekers[column],
                                                                    optimazition_target=self.optimazition_target,
                                                                    matching_condition=self.matching_condition)
                        end = time.time()
                        # print('计算taker权重时间', end - start)
                    else:
                        continue
                else:
                    if vehicles[row - len(takers)].zone == zone:
                        start = time.time()
                        matrix[row, column] = self.calVehiclesWeights(vehicles[row - len(takers)], seekers[column],
                                                                      optimazition_target=self.optimazition_target,
                                                                      matching_condition=self.matching_condition)
                        end = time.time()
                        # print('计算Vehicle权重时间', end - start)
                    else:
                        continue

        # 计算司机选择调度的权重
        for row in range((row_nums - 1)):
            matrix[row, column_nums - 1] = 0

        # 计算乘客选择等待的权重
        for column in range(len(seekers)):
            for row in range(len(takers) + len(vehicles), row_nums):
                matrix[row, column] = \
                    self.calSeekerWaitingWeights(seekers[column],
                                                 optimazition_target=self.optimazition_target)

        end = time.time()
        # print('构造矩阵用时', end-start)
        # print(matrix)
 
        # 匹配
        import time
        start = time.time()
        matcher = KM_method.KM_method(matrix)
        res, weights = matcher.run()
        end = time.time()

        # 匹配成功的字典
        matched_dic = {}
        # 未匹配成功的车辆列表
        unmatched_lis = []
        # 未匹配成功的乘客列表
        unmatched_passengers = []
        for i in range(len(takers)):
            #  第i个taker响应第res[1][i]个订单
            if res[i] >= len(seekers):
                unmatched_lis.append(takers[i].id)
            else:
                # 匹配到新乘客
                matched_dic[takers[i].id] = seekers[res[i]].id
                seekers[res[i]].service_target = 1

        for i in range(len(vehicles)):
            #  第i个taker响应第res[1][i]个订单
            if res[i + len(takers)] >= len(seekers):
                unmatched_lis.append(vehicles[i].id)
            else:
                # 匹配到新乘客
                matched_dic[vehicles[i].id] = seekers[res[i + len(takers)]].id
                seekers[res[i + len(takers)]].service_target = 1

        for i in range(len(seekers)):
            if seekers[i].service_target == 0:
                unmatched_passengers.append(seekers[i].id)

        action = [matched_dic, unmatched_lis, unmatched_passengers ]   
        # print('匹配时间{},匹配成功{},匹配失败{},takers{},vehicles{},demand{},time{}'.
        #       format(end-start, successed, failed, len(takers), len(vehicles), len(seekers), self.time_slot))
        return action

    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        # expected shared distance
        Route, pick_up_distance = self.get_path((seeker.o_x,seeker.o_y),(taker.x,taker.y))

        fifo, distance = self.is_fifo(taker.order_list[0], seeker)

        if fifo:
            shared_distance = self.get_path(
                (seeker.o_x,seeker.o_y), (taker.order_list[0].d_x, taker.order_list[0].d_y) )[1]

            p0_invehicle = pick_up_distance + distance[0]
            p1_invehicle = sum(distance)
            p0_detour = p0_invehicle - taker.order_list[0].shortest_distance
            p1_detour = p1_invehicle - seeker.shortest_distance

        else:
            shared_distance = seeker.shortest_distance
            p0_invehicle = pick_up_distance + sum(distance)
            p1_invehicle = distance[0]
            p0_detour = p0_invehicle - taker.order_list[0].shortest_distance
            p1_detour = p1_invehicle - seeker.shortest_distance

        if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                   p0_detour > self.detour_distance_threshold or
                                   p1_detour > self.detour_distance_threshold):
            # print('detour_distance not pass', detour_distance)
            return self.cfg.dead_value
        else:
            reward = (seeker.shortest_distance + taker.order_list[0].shortest_distance
                      - (p0_invehicle + p1_invehicle - shared_distance) - pick_up_distance) * seeker.delay
            # print('taker reward',reward)
            return reward

    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):

        pick_up_distance = self.get_path(
            (seeker.o_x,seeker.o_y), (vehicle.x , vehicle.y))[1]
        if matching_condition and (pick_up_distance > self.cfg.pickup_distance_threshold or seeker.esdt < 0):
            return self.cfg.dead_value
        else:
            reward = (seeker.esdt - pick_up_distance) * seeker.delay
            # print('vacant vehicle reward',reward)
            return reward

    # 计算乘客选择等待的权重

    def calSeekerWaitingWeights(self, seeker,  optimazition_target):
        if optimazition_target == 'platform_income':
            # 不可行
            return seeker.delay

        else:  # expected shared distance
            gamma = 11.67
            reward = seeker.esds - gamma * seeker.k * 60
            # reward = (seeker.esds) * seeker.delay

            return reward

    def is_fifo(self, p0, p1):

        fifo = [self.get_path((p1.o_x,p1.o_y),(p0.d_x,p0.d_y))[1],
                self.get_path((p0.d_x,p0.d_y),(p1.d_x,p1.d_y))[1] ]

        lifo = [self.get_path((p1.o_x,p1.o_y),(p1.d_x,p1.d_y))[1],
                self.get_path((p1.d_x,p1.d_y),(p0.d_x,p0.d_y))[1] ]

        if sum(fifo) < sum(lifo):
            return True, fifo
        else:
            return False, lifo

    def get_path(self, O, D):
        tmp = self.shortest_path[(self.shortest_path['O'] == O) & (
            self.shortest_path['D'] == D)]
        if tmp['distance'].unique():
            return tmp['distance'].unique()[0]
        else:
            return self.network.get_path(O, D)[0]

    def save_metric(self, path):
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(self.res, tf)

    def save_his_order(self, path):
        dic = {}
        for i in range(len(self.his_order)):
            dic[i] = self.his_order[i]
        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic, tf)
