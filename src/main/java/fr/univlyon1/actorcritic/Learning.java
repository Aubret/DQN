package main.java.fr.univlyon1.actorcritic;

import main.java.fr.univlyon1.environment.ActionSpace;
import main.java.fr.univlyon1.environment.Observation;
import main.java.fr.univlyon1.networks.Approximator;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Learning<A> {
    A getAction(INDArray input);
    void putReward(Double reward);
    Approximator getApproximator();
    ActionSpace<A> getActionSpace();
    void stop();
}
