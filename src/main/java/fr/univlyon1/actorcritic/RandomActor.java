package fr.univlyon1.actorcritic;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.environment.ActionSpace;
import fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class RandomActor<A> implements Learning<A> {
    private long seed ;
    private Random random ;
    private ActionSpace<A> actionSpace ;

    public RandomActor(long seed){
        this.seed = seed ;
        this.random = new Random(seed);
        Nd4j.getRandom().setSeed(seed);
    }
    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public A getAction(INDArray input) {
        INDArray mapping = Nd4j.rand(1,1);
        return this.actionSpace.mapNumberToAction(mapping);
    }

    @Override
    public void putReward(Double reward) {

    }

    @Override
    public Approximator getApproximator() {
        return null;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return this.actionSpace;
    }

    @Override
    public void stop() {

    }
}
