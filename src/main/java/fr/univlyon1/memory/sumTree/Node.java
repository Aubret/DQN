package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;

public class Node<A> implements INode<A> {
    private INode<A> left;
    private INode<A> right ;
    private Double sum ;
    InteractionHistory<A> ih ;

    public Node(InteractionHistory<A> ih){
        this.left = null ;
        this.right = null ;
        this.ih = ih ;
        this.sum = 0.;
    }

    @Override
    public void insert(InteractionHistory<A> ih) {
        this.sum += ih.getErrorValue();
        if(ih.getErrorValue() <= this.ih.getErrorValue()){
            if(this.left == null){
                this.left = new Node<A>(ih);
            }else{
                this.left.insert(ih);
            }
        }else{
            if(this.right == null){
                this.right = new Node<A>(ih);
            }else{
                this.right.insert(ih);
            }
        }

    }
}
