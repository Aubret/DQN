package fr.univlyon1.selfsupervised;

import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.networks.Approximator;

public interface PomdpLearner<A> {
    void step();
    void notify(Observation observation);
    void stop();
}
