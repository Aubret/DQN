package fr.univlyon1.networks;

public interface StateApproximator extends Approximator {
    Object getMemory();
    void setMemory(Object memory);
    StateApproximator clone(); // clônage
    StateApproximator clone(boolean Listener);
}
