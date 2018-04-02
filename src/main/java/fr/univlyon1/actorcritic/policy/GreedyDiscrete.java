package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class GreedyDiscrete implements Policy {

    public GreedyDiscrete(){}

    @Override
    public Integer getAction(INDArray results) {
        return Nd4j.argMax(results).getInt(0);
    }
}
