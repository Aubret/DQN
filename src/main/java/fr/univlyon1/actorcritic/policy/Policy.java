package fr.univlyon1.actorcritic.policy;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Policy{
    Object getAction(INDArray results);
}
