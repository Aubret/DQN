package fr.univlyon1.agents;

import fr.univlyon1.environment.space.Observation;
import fr.univlyon1.environment.space.SpecificObservation;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Main class
 * @param <A>
 */
public interface AgentRL<A> {
    /**
     * Get a simple reward
     * @param reward
     * @param observation
     * @param dt
     * @return
     */
    A control(Double reward,Observation observation,Double dt);

    /**
     * Get a reward a a SMDP, reward map the reward to the time is arrived since the last action.
     * @param reward
     * @param evaluation
     * @param observation
     * @param dt
     * @return
     */
    A control(HashMap<Double,Double> reward, ArrayList<Double> evaluation, Observation observation, Double dt);

    /**
     * Learning for self-supervied learning, not used anymore
     * @param observation
     */
    void notify(Observation observation);

    /**
     * Stop the learning process
     */
    void stop();
}
