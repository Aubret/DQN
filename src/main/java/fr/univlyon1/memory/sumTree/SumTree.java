package fr.univlyon1.memory.sumTree;

import fr.univlyon1.memory.prioritizedExperienceReplay.InteractionHistory;

public class SumTree<A> extends Node<A> {

    private Node<A> root ;

    public SumTree(){
        super(null,null);
    }
    public void remove(Node<A> node,Node<A> attach){
        this.root = attach ;
        if(attach != null)
            attach.setParent(this);
    }

    public Double getTotalSum(){
        if(this.root != null)
            return this.root.getSum() ;
        else
            return 0.;
    }



    public InteractionHistory<A> getInteractionUp(double value){
        if(this.root == null)
            return null ;
        return this.root.getUp(value).getIh();
    }

    public Node<A> removeFirst(){
        return this.root.removeFirst();
    }

    public Node<A> removeLast(){
        return this.root.removeLast();
    }

    public InteractionHistory<A> getFirst(){
        return this.removeFirst().getIh() ;
    }

    public InteractionHistory<A> getLast(){
        return this.removeLast().getIh() ;
    }

    public void insert(InteractionHistory<A> ih){
        if(this.root != null) {
            this.root.insert(ih);
        }else{
            this.root = new Node<A>(ih,this);
            //System.out.println("ROOT "+this.root.getIh().getErrorValue());
        }
    }

    public int size(){
        return this.root == null ? 0 : this.root.getNumberDescendant()+1 ;
    }
}
