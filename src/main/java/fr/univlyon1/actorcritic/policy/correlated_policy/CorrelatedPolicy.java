package fr.univlyon1.actorcritic.policy.correlated_policy;

import akka.serialization.Serialization;
import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.learning.Informations;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface CorrelatedPolicy<A> extends Policy<A> {
    Object getAction(INDArray results, Informations information);

}
