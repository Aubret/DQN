package fr.univlyon1.actorcritic.policy;

import fr.univlyon1.learning.Informations;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Policy<A>{
    Object getAction(INDArray results,Informations information);
}
