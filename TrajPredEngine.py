from ignite.engine import Engine

class IgniteEngine:
    """
    An abstract class representing a pytorch ignite engine. 
    Enables definition of a variety of training / evaluation functions for different
    trajectory prediction algorithms

    TODO: standardize arguments
    """
    def train_batch(self, engine, batch):
        raise NotImplementedError

    def eval_batch(self, engine, batch):
        raise NotImplementedError

    def makeTrainer(self):
        """ Creates a trainer and evaluator and attaches event handlers
        """
        #return Engine(train_batch) 
        raise NotImplementedError