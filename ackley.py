import math

def power(my_list):
  return [x**2 for x in my_list]

def cos_2_pi(my_list):
  return [math.cos(2 * math.pi * x) for x in my_list]

def ackley_function(fenotipos):
  sum_1 = sum(power(fenotipos))
  sum_2 = sum(cos_2_pi(fenotipos))

  termo_1 = -20 * math.exp(-0.2 * math.sqrt(sum_1/30))
  termo_2 = -math.exp(sum_2/30)

  return termo_1 + termo_2 + 20 + math.e