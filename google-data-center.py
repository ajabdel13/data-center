import numpy as np
import random
import tensorflow as tf

class DatacenterEnv(object):
    
    def __init__(self, optimal_temperature = (16, 24), initial_month = 0,initial_users= 20 , initial_rate_data= 30):
        
        self.monthly_atmospheric_temperature = [6, 12, 23, 4, 6, 9, 12, 25, 18, 11, 7]
        self.initial_month = initial_month
        self.optimal_temperature = optimal_temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[initial_month]
        self.initial_users = initial_users
        self.max_temperature = 80
        self.min_temperature = -20
        self.max_users = 100
        self.min_users = 20
        self.max_update_users = 5
        self.initial_rate_data = initial_rate_data
        self.max_rate_data = 500
        self.min_rate_data = 30
        self.max_update_data = 10
        self.current_users = initial_users
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature= 1.25 * self.atmospheric_temperature + 1.95 * self.current_users + 1.95 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0]+ self.optimal_temperature[1])/2
        self.total_energy_ai = 0
        self.total_energy_noai = 0
        self.reward = 0
        self.game_over = 0
        self.train = True

    def update_env_ai(self, energy_ai, direction, month):
        
        #energy with no ai
         energy_noai = 0
        if (self.temperature_noai > self.optimal_temperature[1]):
            enregy_noai = self.optimal_temperature[1] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[1]
        elif (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.temperature_noai - self.optimal_temperature[0]
            self.temperature_ai = self.optimal_temperature[0]

        self.reward = energy_noai - energy_ai
        self.reward = 1e-3 * self.reward
        # atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        # current users
        self.current_users += np.random.randint(-self.max_update_users, self.max_update_users)
        if (self.current_users> self.max_users):
           self.current_users = self.max_users
        elif (self.current_users < self.min_users):
           self.current_users = self.min_users
        # current rate data 
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
           self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
           self.current_rate_data = self.min_rate_data

        past_intrinsic_temperature = self.intrinsic_temperature
        self.intrinsic_temperature = 1.25 * self.atmospheric_temperature + 1.95 * self.current_users + 1.95 * self.current_rate_data
        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature

        if (direction = -1):
           delta_temperature_ai = -energy_ai
        else:
           delta_temperature_ai = energy_ai
        
        self.temperature_ai += delta_temperature_ai + delta_intrinsic_temperature
        self.temperature_noai = delta_intrinsic_temperature

        if (self.temperature_ai > self.max_temperature):
           if (self.train== 1):
               self.game_over = 1
           else:
               self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1] 
               self.temperature_ai = self.optimal_temperature[1]

        if (self.temeperature_ai < self.min_temperature):
           if (self.train == 1):
              self.game_over = 1
           else :
              self.total_energy_ai += self.optimal_temperature[0] - self.temeperature_ai
              self.temeperature_ai = self.optimal_temperature[0]

        self.total_energy_ai += energy_ai
        self.total_energy_noai + = enregy_noai

        scaled_temperature_ai = (self.temeperature_ai - self.min_temperature)/(self.max_temperature-self.min_temperature)
        scaled_current_users =  (self.current_users - self.min_users)/(self.max_users-self.min_users)
        scaled_current_rate_data = (self.current_rate_data- self.min_rate_data)/(self.max_rate_data-self.min_rate_data)
        next_state = np.matrix([scaled_temperature_ai, scaled_current_users,scaled_current_rate_data])
        return next_state, self.reward, self.game_over
    
    def reset (self, month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperature[month]
        self.initial_users = initial_users
        self.current_users = initial_users
        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature= 1.25 * self.atmospheric_temperature + 1.95 * self.current_users + 1.95 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0]+ self.optimal_temperature[1])/2
        self.total_energy_ai = 0
        self.total_energy_noai = 0
        self.reward = 0
        self.game_over = 0
        self.train = 1

    def Observe():
        scaled_temperature_ai = (self.temeperature_ai - self.min_temperature)/(self.max_temperature-self.min_temperature)
        scaled_current_users =  (self.current_users - self.min_users)/(self.max_users-self.min_users)
        scaled_current_rate_data = (self.current_rate_data- self.min_rate_data)/(self.max_rate_data-self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai, scaled_current_users,scaled_current_rate_data])
        return current_state, self.reward, self.game_over

Datacenter = DatacenterEnv()

state_size = 3
action_size = 5
learning_rate = 0.9
max_size = 10000
batch_size = 512
pretrain_length = 1000
action_boundry = action_size/2
possible_action = [0,1,2,3,4]
step_max = 1000

#Create The Brain 
class Brain:

    def __init__(self, state_size, action_size, learning_rate, Training, name="DQNbrain"):
       self.state_size = state_size
       self.action_size = action_size
       self.learning_rate = learning_rate
       self.Training = False

       with tf.variable_scope(name):

           self.input_ = tf.placeholder(tf.float32, [None , state_size], name="input")
           self.output_ = tf.placeholder(tf.float32, [None, 5], name="output")
           self.target_Q = tf.placeholder(tf.float32,[None], name="Target")

           self.fc1 =tf.layers.dense(inputs=state_size, 
                                     units= 500,
                                     activation=tf.nn.elu,
                                     kernel_initializer= tf.contrib.layers.xavier_initializer(),
                                     name="fc1")
                                
           self.fc1 = tf.layers.dropout(self.fc1, rate= 0.2, training= True)

           self.output = tf.layers.dense(inputs=500, 
                                     units= 5,
                                     activation=None,
                                     kernel_initializer= tf.contrib.layers.xavier_initializer(),
                                     name="fc2")
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.output_))
            self.loss = tf.reduce_mean(tf.square(self.Q ,self.target_Q))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

Training = True
tf.reset_default_graph()
DQNetwork = DQNetwork(state_size, action_size, Training)

# Experience Memory

class Memory:
    
    def __init__(self, max_size):
       self.buffer = deque(maxlen = max)

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > max_size
           del self.buffer[0]
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                  size= batch_size,
                                  replace = False )
        return [ self.buffer[i] for i in range index]

memory = Memory(10000)


#Handle the problem of empty Memory
DataCenter.reset()
for i in range (pretrain_length):
     
     state = Datacenter.Observe()
     month=np.random.randint(0,12)
     action = np.random.randint(0,5)
     if (action-action_boundry)< 0 :
         direction = -1
         energy_ai = abs(action-action_boundry) * 1.5
     else:
         direction = 1
         energy_ai = abs(action - action_boundry) * 1.5
     next_state, self.game_over, self.reward = update_env_ai(self, energy_ai, direction, month)

     if self.game_over:
          next_state = np.zeros(state_size)
          memory.add((state, next_state, self.reward, action, self.game_over))
          DataCenter.reset()
          state= Datacenter.Observe()
     else :
          memory.add((state, next_state, self.reward, action, self.game_over))
          state = next_state

#Setup TensorBoard 

writer = tf.summary.FileWriter("/Tensorboard/dqn/1")
tf.summary.Scalar("Loss", DQNetwork.loss)
write_op = tf.summary_merge_all

#Train the model:
def predict(self, explore_start, explore_stop, decay_rate, decay_step, state, possible_action):

         exp_exp_tradeoff = np.random.randint()
         probabilitie_exp = explore_stop + (explore_start - explore_stop) * (-decay_rate * decay_step)
         if (exp_exp_tradeoff < probabilitie_exp):
             action = possible_action
         else:

            Qs = sess.run{Brain.output, feed_dict:{ Brain.input_: state_size}}
            choice = np.argmax(Qs)
            action = possible_action[choice]

         return action, probabilitie_exp

saver = tf.train.Saver()

if(training= True):
     with tf.Session() as sess:
           sess.run(tf.global_variable_initializer())
           decay_step = 0
           DatacenterEnv.reset(month)

           for episode in range(total_episode):
               step = 0
               episode_reward = []
               state = DatacenterEnv.Observe()

               while (step < step_max):
                      
                      
                      step += 1
                      action = predict(explore_start, explore_stop, decay_rate, decay_step, state, possible_action)
                      
                      if ( action - action_boundry< 0):
                          direction = -1
                          energy_ai = abs(action - action_boundry) * 1.5
                      else:
                          direction = 1
                          energy_ai = abs(action - action_boundry) * 1.5
                      month = np.random.randint(0,12)
                      next_state, self.reward, self.game_over= update_env_ai(self, energy_ai, direction, month)
                      episode_reward.append(self.reward)
                      
                      if self.game_over:
                            next_state = np.zeros(state_size)
                            memory.add((state, action,self.reward, next_state, self.game_over))
                            step = step_max
                            total_reward = np.sum(episode_reward)
                           
                            print("Episode:{}".format(episode))
                            print("total_reward:{}".format(total_reward))
                            print("Training loss:{}".format(loss))
                            print("Explore P:{}".format(probabilitie_exp))

                      else:
                           next_state, self.reward, self.game_over= update_env_ai(self, energy_ai, direction, month)
                           memory.add((state, action, next_state, self.reward, self.game_over))
                           state = next_state


                      batch = memory.sample(batch_size)
                      state_mb = np.array[each[0] for each in batch]
                      action_mb = np.array[each[1] for each in batch]
                      next_state_mb = np.array[each[2] for each in batch]
                      reward_mb = np.array[each[3] for each in batch]
                      game_over_mb = np.array[each[4] for each in batch]

                      target_Qs_batch = []

                      Qs_next_batch = sess.run(Brain.output, feed_dict= {Brain.input_:next_state_mb})

                      for i in range (0, len(batch)):
                          terminal = game_over_mb[i]

                          if terminal :
                                 target_Qs_batch.append(reward_mb[i])
                          else :
                             target = reward_mb[i] + gamma * np.max(Qs_next_batch[i])
                             target_Qs_batch.append(target)

                          target_mb = np.array([each for each in target_Qs_batch])

                      loss , _ = sess.run([Brain.loss Brain.optimizer],
                                               feed_dict={Brain.input_: state_mb
                                                          Brain.target_Q: target_mb,
                                                          Brain.action_: action_mb})

                      summary = sess.run(write_op, feed_dict={Brain.input_: state_mb
                                                          Brain.target_Q: target_mb,
                                                          Brain.action_: action_mb})  
                      writer.add_summary(summary, episode)
                      writer.flush()

                      if episode % 5 == 0:
                           save_path = saver.save(sess,"./models/model.ckpt")
                           print("Model Saved")
#Test Model:
with tf.Session() as sess:

     DataCenter= Datacenterenv()
     total_score = 0
     saver.restore(sess, "./models/model.ckpt")

     for timestep in range (10000):

         self.game_over = False
         DataCenter.reset()

         state, _, _ = DataCenter.Observe()

         While (not self.game_over):
              Qs = sess.run(Brain.output, feed_dict={Brain.input_: state})

              choice = np.argmax(Qs)
              action = possible_action[int(choice)]
              if ( action - action_boundry< 0):
                  direction = -1
                  energy_ai = abs(action - action_boundry)*1.5
              else:
                  direction = 1
                  energy_ai = abs(action - action_boundry)*1.5
                  
              next_state, reward, game_over = Datacenter.update_env_ai(energy_ai,  direction, timestep/(30*24*60))
              state = next_state
print("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))               
            
            










                       




          



      

     


     

















    




            

        
        



                
