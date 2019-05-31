from ignite.engine import Engine

class IgniteEngine:
    """
    An abstract class representing a pytorch ignite engine. 
    Enables definition of a variety of training / evaluation functions for different
    trajectory prediction algorithms
    """
    def train_batch(self, engine, batch):
        raise NotImplementedError

    def eval_batch(self, engine, batch):
        raise NotImplementedError

    def getTrainer(self):
        #return Engine(train_batch) 
        raise NotImplementedError