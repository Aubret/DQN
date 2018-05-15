package fr.univlyon1.networks;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

public interface StateApproximator extends Approximator {
    Object getMemory();
    Object getSecondMemory();
    INDArray forwardLearn(INDArray input, INDArray labels, int number, INDArray mask, INDArray maskLabels);
    void setMemory(Object memory);
    StateApproximator clone(); // clônage
    StateApproximator clone(boolean Listener);
}
