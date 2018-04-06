package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;

public class SumTree<A> {

    private Node root ;

    public SumTree(){
        //this.root =  new Node(0.);
    }

    public void insert(InteractionHistory<A> ih){
        this.root.insert(ih);
    }
}
