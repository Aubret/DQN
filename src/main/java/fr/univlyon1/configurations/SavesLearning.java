package fr.univlyon1.configurations;

import fr.univlyon1.agents.AgentDRL;

import java.io.FileWriter;
import java.io.PrintWriter;

public class SavesLearning {

    private PrintWriter saveQ ;


    public SavesLearning(){
        try {
            //FileWriter fw = new FileWriter("sim/arthur/continuous_rewards_baseline.csv");
            FileWriter fw = new FileWriter("sim/arthur/save/"+AgentDRL.getFilename()+".csv");
            //FileWriter fw = new FileWriter("sim/arthur/results/a6_baseline.csv");
            this.saveQ = new PrintWriter(fw);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void add(double Q){
        this.saveQ.println(AgentDRL.getCount() +";"+Q);
    }

}
