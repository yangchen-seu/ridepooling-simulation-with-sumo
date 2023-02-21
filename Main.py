
import Simulation as sm
import Config



class Main():

    def __init__(self,cfg ) -> None:
        self.simu = sm.Simulation(cfg)
        self.test_episodes = 1
 
 
    def run(self):
        rewards = []
        for i_ep in range(self.test_episodes):
            # print('episode:', i_ep)
            ep_reward = 0 # 一个回合的奖励
            self.simu.reset()
            while True:
                reward, done = self.simu.step()
                # print(self.simu.time)
                # print('reward', reward)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)

            # print('rewards',rewards,smooth_rewards,'smooth_rewards')
        res = self.simu.res
        print('waitingTime:',res['waitingTime'])
        print('detour_distance:',res['detour_distance'])
        print('pickup_time:',res['pickup_time'])
        print('shared_distance:',res['shared_distance'])
        print('total_ride_distance:',res['total_ride_distance'])
        print('saved_ride_distance:',res['saved_ride_distance'])
        print('platform_income:',res['platform_income'])
        print('response_rate:',res['response_rate'])
        print('carpool_rate:',res['carpool_rate'])

        
        # self.plot_metrics(cfg, metric = self.simu.detour_distance, metric_name = 'detour_distance')
        # self.plot_metrics(cfg, metric =self.simu.traveltime, metric_name = 'traveltime')
        # self.plot_metrics(cfg, metric = self.simu.waitingtime, metric_name = 'waiting_time')
        # self.plot_metrics(cfg, metric = self.simu.pickup_time, metric_name = 'pickup_time')
        # self.plot_metrics(cfg, metric = self.simu.platform_income, metric_name = 'platform_income')
        # self.plot_metrics(cfg, metric = self.simu.shared_distance, metric_name = 'shared_distance')
        
        return res

    # 绘图
    def test_plot(self,rewards, smoothed_rewards, algo, env_name):
 
        self.plot_rewards(rewards, smoothed_rewards,  algo, env_name)
 
 
    # 绘图
    def plot_rewards(self, cfg, rewards,smoothed_rewards, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("reward of {} for {}".format(algo,env_name))
        plt.xlabel('epsiodes')
        plt.plot(rewards,label='rewards')
        plt.plot(smoothed_rewards,label='smoothed rewards')
        plt.legend()
        # plt.savefig('output\\rewards.png')
        plt.savefig('output\\{}/rewards.png'.format(cfg.optimazition_target))      

     # 绘图
    def plot_metrics(self,cfg, metric,  metric_name, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("{} of {} for {}".format(metric_name, algo,env_name))
        plt.xlabel('orders')
        plt.plot(metric,label=metric_name)
        plt.legend()
        plt.savefig('output\\{}/{}.png'.format(cfg.optimazition_target, metric_name))       



# ratios = [100/25,100/50,100/75]
# for ratio in ratios:
#     cfg = Config.Config()
#     cfg.order_driver_ratio = ratio
#     print('ratio:',cfg.order_driver_ratio)
#     import time
#     start = time.time()
#     ma = Main(cfg)
#     ma.run() 
#     end = time.time()
#     print('执行时间{},order_driver_ratio:{}'.format(end - start, cfg.order_driver_ratio))
#     print('ratio:',ma.simu.vehicle_num / len(ma.simu.order_list))


import os
files = os.listdir('input/orders/')
print(files)
dic = {}
import pickle
files = [  '2017-05-15.csv', '2017-05-16.csv', '2017-05-17.csv', '2017-05-18.csv', '2017-05-19.csv', '2017-05-20.csv', '2017-05-21.csv']
with open('output/results.pkl', "wb") as tf:
    for file in files:
        print(file)
        cfg = Config.Config()
        print('ratio:',cfg.order_driver_ratio)
        cfg.date = file.split('.')[0]
        cfg.order_file = 'input/orders/'+ file
        import time
        start = time.time()
        ma = Main(cfg)
        ma.run() 
        end = time.time()
        print('file:{},执行时间:{}'.format(file, end - start))
        dic[file.split('.')[0]] = ma
    pickle.dump(dic, tf)