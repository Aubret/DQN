package fr.univlyon1.agents;

import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.SpecificObservation;

import java.util.ArrayList;
import java.util.HashMap;

public interface AgentRL<A> {
    A control(Double reward,Observation observation,Double dt);
    A control(HashMap<Double,Double> reward, ArrayList<Double> evaluation, Observation observation, Double dt);
    void notify(Observation observation);
    void stop();
}
