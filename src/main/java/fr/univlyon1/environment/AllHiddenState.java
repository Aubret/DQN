package fr.univlyon1.environment;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

@Getter
public class AllHiddenState {

    ArrayList<INDArray[]> timeStates ;
    ArrayList<INDArray[]> prevActs ;

    public AllHiddenState(ArrayList<INDArray[]> timeStates, ArrayList<INDArray[]> prevActs){
        this.timeStates = timeStates ;
        this.prevActs = prevActs ;
    }
}
