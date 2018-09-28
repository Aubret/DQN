package fr.univlyon1.environment.space;

import fr.univlyon1.environment.interactions.Replayable;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface SpecificObservation<A> extends Observation,Comparable,Replayable<A> {

    long getId();
    boolean hasAlreadySent();
    Double getOrderedNumber();
    INDArray getLabels();
}
