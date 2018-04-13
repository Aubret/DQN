package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Node<A>{
    private Node<A> left;
    private Node<A> right ;
    private Double sum ;
    private InteractionHistory<A> ih ;

    private Node<A> parent;

    public Node(InteractionHistory<A> ih,Node parent){
        this.left = null ;
        this.right = null ;
        this.ih = ih ;
        this.sum = ih.getErrorValue();
        this.parent = parent ;
    }

    public void insert(InteractionHistory<A> ih) {
        this.sum += ih.getErrorValue();
        if(ih.getErrorValue() <= this.ih.getErrorValue()){
            if(this.left == null){
                this.left = new Node<A>(ih,this);
            }else{
                this.left.insert(ih);
            }
        }else{
            if(this.right == null){
                this.right = new Node<A>(ih,this);
            }else{
                this.right.insert(ih);
            }
        }
    }

    public void remove(Node node,Node attach){
        if(this.left == node) {
            this.left = attach;
        }
        if(this.right == node)
            this.right = attach ;
    }

    public InteractionHistory<A> removeFirst() {
        if(this.left ==null){
            this.parent.remove(this,this.right);
            return this.ih ;
        }

        InteractionHistory<A> i = this.left.removeFirst();
        this.sum =( this.left!= null ? this.left.getSum() :0. )+ ( this.right!= null ? this.right.getSum():0. );
        return i ;
    }
}
