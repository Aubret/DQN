package fr.univlyon1.networks;

public interface StateApproximator extends Approximator {
    Object getMemory();
    void setMemory(Object memory);
    void setBackpropNumber(int backpropNumber);
    void setForwardNumber(int forwardNumber);
    StateApproximator clone(); // clônage
    StateApproximator clone(boolean Listener);
}
