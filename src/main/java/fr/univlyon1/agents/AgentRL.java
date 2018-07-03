package fr.univlyon1.agents;

import fr.univlyon1.environment.space.Observation;

import java.util.HashMap;

public interface AgentRL<A> {
    A control(Double reward,Observation observation,Double dt);
    A control(HashMap<Double,Double> reward, Observation observation, Double dt);
    void stop();
}
