package fr.univlyon1.learning;

import akka.serialization.Serialization;
import fr.univlyon1.environment.space.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Algorithm<A> {
    INDArray behave(INDArray input);
    void step(Observation observation, A action, Double time);
    void evaluate(Observation input, Double reward, Double time);
    void epoch();
    void learn();
    Informations getInformation();

}
