
class Arm:
  """
  An arm object with rewards data
  """
  def __init__(self, data, idx):
      self.data = data.values[:, idx]
      self.length = len(data)
      self.idx = idx
      self.expectation = sum(self.data)/self.length/10

  def draw(self, n):
      reward = self.data[n]
      reward = reward/10

      return reward