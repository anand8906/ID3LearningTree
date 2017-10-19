import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;


import weka.core.Instance;
import weka.core.Instances;


public class ID3Driver {
	
	public static List<Integer> findCandidateSplits(List<Integer> records, Instances data)
	{
		List<Integer> candidateSplits = new ArrayList<>();
		for(int i = 0; i < data.numAttributes()-1; i++)
		{
			
			String prev = null;
			for(int j = 0; j < records.size(); j++)
			{
				if(prev == null) prev = data.instance(records.get(j)).toString(i);
				else 
				{
					if(prev.equals(data.instance(records.get(j)).toString(i)))
					  continue;
					else
					{
					  candidateSplits.add(i);
					  break;
					}
						
				}
			}
		}
		return candidateSplits;
	}
	
	private static double calculateProbabilityOfY(List<Integer> records, Instances data) {
	  String prev = null;
	  double count = 0.0;
	  double totalRecords = records.size();
	  for(int i = 0; i < records.size();i++)
	  {
         if(prev == null) {
        	     prev = data.instance(records.get(i)).stringValue(data.classAttribute());
         }
         else 
         {
        	   if(!(prev.equals(data.instance(records.get(i)).stringValue((data.classAttribute())))))
        		   count++;
         }
	  }
	  return count/totalRecords;
	}
	
	private static double calculateEntropy(List<Integer> records, Instances data) 
	{
		double pOfY = calculateProbabilityOfY(records, data);
		if(pOfY != 0 && pOfY != 1) 
	  	  return -1 * (pOfY*(Math.log(pOfY)/Math.log(2)) + (1-pOfY)*(Math.log(1-pOfY)/Math.log(2)));
		else
		  return 0.0;
	}
	
	private static double calculateEntropyOfRandomVariable(double probability)
	{
		return -1*((probability*(Math.log(probability)/Math.log(2))) 
				   + ((1-probability)*(Math.log(1-probability)/Math.log(2))));
	}
	
	private static NumericSplit calculateNumericConditionalEntropy(Integer featureSplit, List<Integer> records, Instances data)
	{
		NumericSplit split = new NumericSplit();
		split.setSplitPosition(featureSplit);
		Map<Double, Set<String> > S = new TreeMap<Double, Set<String> >();
		List<Double> splits = new ArrayList<>();
		double minConditionalEntropy = Double.MAX_VALUE;
		for(Integer i : records)
		{
			Double key = data.instance(i).value(featureSplit);
			String classValue = data.instance(i).toString(data.classAttribute());
			if(S.containsKey(key))
				S.get(key).add(classValue);
			else
			{
				Set<String> value = new HashSet<>();
				value.add(classValue);
				S.put(key, value);
			}
		}
		
	    Iterator<Double> keys = S.keySet().iterator();
	    List<Double> keyList= new ArrayList<>();
	    while(keys.hasNext())
	    	  keyList.add(keys.next());
	    for(int i = 0; i < keyList.size()-1; i++)
	    {
	    	    if((S.get(keyList.get(i)).size() == 2) || (S.get(keyList.get(i+1)).size() == 2))
	    	    	  splits.add((keyList.get(i) + keyList.get(i+1))/2);
	    	    else
	    	    {
	    	    	    if(!(S.get(keyList.get(i)).iterator().next().equals(S.get(keyList.get(i+1)).iterator().next())))
	    	    	    	    splits.add((keyList.get(i) + keyList.get(i+1))/2);
	    	    }
	    	
	    }
	    
	    for(Double threshold : splits)
	    {
			double currentEntropy = Double.MAX_VALUE;
			double lessThanPositiveCount = 0;
			double lessThanCount = 0;
			double greaterThanPositiveCount  = 0;
			double greaterThanCount = 0;
			
			for(int i = 0; i < records.size(); i++)
			{
			  if(data.instance(records.get(i)).value(featureSplit) > threshold)
			  {
				  greaterThanCount++;
				  if(data.instance(records.get(i)).toString(data.classAttribute()).equals("positive"))
				  {
					  greaterThanPositiveCount++;
				  }
			  }
			  else
			  {
				 lessThanCount++;
			     if(data.instance(records.get(i)).toString(data.classAttribute()).equals("positive"))
			     {
			    	     lessThanPositiveCount++;
			     }
			  }
			}
			
			double lessThanEntropy = 0.0;
			double greaterThanEntropy = 0.0;
			if(lessThanCount != 0)
			{
				double lessThanPositiveProb = lessThanPositiveCount/lessThanCount;
				if(lessThanPositiveProb != 0 && lessThanPositiveProb != 1)
				{
					lessThanEntropy = ((lessThanCount/records.size())*calculateEntropyOfRandomVariable(lessThanPositiveCount/lessThanCount)) ;
				}
			}
			if(greaterThanCount != 0)
			{
				
				double greaterThanPositiveProb = greaterThanPositiveCount/greaterThanCount;
				if(greaterThanPositiveProb != 0 && greaterThanPositiveProb != 1)
				{
				    greaterThanEntropy = ((greaterThanCount/records.size())*calculateEntropyOfRandomVariable(greaterThanPositiveCount/greaterThanCount));
				}
			}
			currentEntropy = greaterThanEntropy + lessThanEntropy;
			if(minConditionalEntropy > currentEntropy)
			{
				split.setThreshold(threshold);
				minConditionalEntropy = currentEntropy;
			}
	    }
	    split.setSplitEntropy(minConditionalEntropy);
	    return split;
	}
	
	private static NominalSplit calculateNominalConditionalEntropy(Integer featureSplit, List<Integer> records, Instances data)
	{
		NominalSplit split = new NominalSplit();
		split.setSplitPosition(featureSplit);
		
		double runningConditionalEntropy = 0;
		for(Object feature : findFeatureRange(records, data, featureSplit))
		{
			double countGivenFeatureValue = 0;
			double countGivenFeaturePositive = 0;
			for(Integer i : records)
			{
			  if(data.instance(i).toString(data.attribute(featureSplit)).equals((String)feature))
			  {
				  countGivenFeatureValue++;
				  if(data.instance(i).toString(data.classAttribute()).equals("positive"))
				  {
					  countGivenFeaturePositive++;
				  }
			  }
			}
			if(countGivenFeatureValue != 0)
			{
				double probability = countGivenFeaturePositive/countGivenFeatureValue;
				if(probability != 1 && probability != 0)
			       runningConditionalEntropy += (countGivenFeatureValue/records.size())*calculateEntropyOfRandomVariable(countGivenFeaturePositive/countGivenFeatureValue);
			}
			split.getSplitValues().add((String)feature);
		}
		split.setSplitEntropy(runningConditionalEntropy);
		return split;
	}
	
	public static Split findBestSplit(List<Integer> featureSplits, List<Integer> records, Instances data)
	{
		Split bestSplit = null;
		Double maxInformationGain = Double.NEGATIVE_INFINITY;
		double entropyOfY = calculateEntropy(records, data);
		for(int i = 0; i < featureSplits.size();i++)
		{
		  if(data.attribute(featureSplits.get(i)).isNominal())
		  {
		      NominalSplit split = calculateNominalConditionalEntropy(featureSplits.get(i), records, data);
		      double informationGain = entropyOfY - split.getSplitEntropy();
			  if((informationGain > 0) && (maxInformationGain < informationGain))
			  {
				maxInformationGain = informationGain;
				bestSplit = split;
			  }
		  }
		  else
		  {
			  NumericSplit split = calculateNumericConditionalEntropy(featureSplits.get(i), records, data);
		      double informationGain = entropyOfY - split.getSplitEntropy();
			  if((informationGain >= 0) && (maxInformationGain < informationGain))
			  {
				maxInformationGain = informationGain;
				bestSplit = split;
			  }
		  }
		}
		return bestSplit;
	}
	
	public static void findDecisionTree(Instances data, List<Integer> records, TreeNode root, int m)
	{
		String prev = null;
		Integer diffClassCt = 0;
		
		//if all the records belong to the same class, implied that we are at a leaf node.
		for(Integer i : records)
		{
			if(prev == null) prev = data.instance(i).toString(data.classAttribute());
			else
			{
			  if(!(data.instance(i).toString(data.classAttribute()).equals(prev)))
				diffClassCt++; 
			}
		}
		if(diffClassCt == 0 && records.size() != 0) 
		{
			TreeNode node = new TreeNode();
			node.setSplitValue(data.instance(records.get(0)).stringValue(data.classAttribute()));
			root.getChildren().add(node);
			return;
		}
		
		//if number of records are less than passed m
		if(records.size() < m)
		{
			TreeNode node = new TreeNode();
			node.setParent(root);
			if(root.getPositiveCount() > root.getNegativeCount())
				node.setSplitValue("positive");
			else if (root.getPositiveCount() < root.getNegativeCount())
				node.setSplitValue("negative");
			else 
			{
				if(root.getParent().getPositiveCount() != null && root.getParent().getNegativeCount() != null)
				{
			      if(root.getParent().getPositiveCount() > root.getParent().getNegativeCount())
				      node.setSplitValue("positive");
			      else
				      node.setSplitValue("negative");
				}
				else if(root.getParent().getPositiveCount() == null && root.getParent().getNegativeCount() != null)
				{
					node.setSplitValue("negative");
				}
				else if (root.getParent().getPositiveCount() != null && root.getParent().getNegativeCount() == null)
				{
					node.setSplitValue("positive");
				}
				else
					node.setSplitValue("positive");
				  
			}
			root.getChildren().add(node);
			/*
			if(records.size() == 8) System.out.println("Caught in size < m!!");
			TreeNode node = new TreeNode();
			int positiveCount = 0;
			int negativeCount = 0;
			for(int i = 0; i < records.size();i++)
			{
				if(data.instance(i).stringValue(data.classAttribute()).equals("positive"))
					positiveCount++;
				else
					negativeCount++;
			}
			if(positiveCount > negativeCount)
				node.setSplitValue("positive");
			else
				node.setSplitValue("negative");
			root.getChildren().add(node);
			*/
			return;
				
		}
		
		List<Integer> featureSplits; 
		featureSplits = findCandidateSplits(records, data);
		Split bestSplit = findBestSplit(featureSplits, records,data);
		if(bestSplit == null)
		{
			TreeNode node = new TreeNode();
			int positiveCount = 0;
			int negativeCount = 0;
			for(int i = 0; i < records.size();i++)
			{
				if(data.instance(i).stringValue(data.classAttribute()).equals("positive"))
					positiveCount++;
				else
					negativeCount++;
			}
			if(positiveCount > negativeCount)
				node.setSplitValue("positive");
			else
				node.setSplitValue("negative");
			node.setParent(root);
			root.getChildren().add(node);
			return;
		}
		
		if(data.attribute(bestSplit.getSplitPosition()).isNumeric())
		{
			ClassCounts classCtsLeft = new ClassCounts();
			List<Integer> recordSplitLeft = findRecordSplit(records, data, bestSplit.getSplitPosition(), ((NumericSplit)bestSplit).getThreshold(), true, classCtsLeft);
			//root.setSplitIndex(bestSplit.getSplitPosition());
			TreeNode currLeftNode = new TreeNode();
			currLeftNode.setSplitValue(((NumericSplit)bestSplit).getThreshold());
			currLeftNode.setOperator("<=");
			currLeftNode.setSplitIndex(bestSplit.getSplitPosition());
			currLeftNode.setNegativeCount(classCtsLeft.getNegativeCt());
			currLeftNode.setPositiveCount(classCtsLeft.getPositiveCt());
			currLeftNode.setParent(root);
			root.getChildren().add(currLeftNode);
			findDecisionTree(data, recordSplitLeft, currLeftNode, m);
			
			ClassCounts classCtsRight = new ClassCounts();
			List<Integer> recordSplitRight = findRecordSplit(records, data, bestSplit.getSplitPosition(), ((NumericSplit)bestSplit).getThreshold(), false, classCtsRight);
			//root.setSplitOn(data.attribute(bestSplit.getSplitPosition()).toString());
			TreeNode currRightNode = new TreeNode();
			currRightNode.setSplitValue(((NumericSplit)bestSplit).getThreshold());
			currRightNode.setOperator(">");
			currRightNode.setSplitIndex(bestSplit.getSplitPosition());
			currRightNode.setNegativeCount(classCtsRight.getNegativeCt());
			currRightNode.setPositiveCount(classCtsRight.getPositiveCt());
			currRightNode.setParent(root);
			root.getChildren().add(currRightNode);
			findDecisionTree(data, recordSplitRight, currRightNode, m);
		}
		else
		{
			for(Object featureValue : ((NominalSplit)bestSplit).getSplitValues())
			{
				ClassCounts classCts = new ClassCounts();
				List<Integer> recordSplit = findRecordSplit(records, data, bestSplit.getSplitPosition(), featureValue, true, classCts);
				//root.setSplitOn(data.attribute(bestSplit.getSplitPosition()).toString());
				TreeNode currNode = new TreeNode();
				currNode.setSplitValue(featureValue);
				currNode.setSplitIndex(bestSplit.getSplitPosition());
				currNode.setOperator("=");
				currNode.setNegativeCount(classCts.getNegativeCt());
				currNode.setPositiveCount(classCts.getPositiveCt());
				currNode.setParent(root);
				findDecisionTree(data, recordSplit, currNode, m);
				root.getChildren().add(currNode);
			}
		}
	}

	private static List<Integer> findRecordSplit(List<Integer> records, Instances data, Integer bestSplit, Object featureValue, boolean lessThan, ClassCounts classCts) {
		List<Integer> split = new ArrayList<Integer>();
		Integer positiveCt = 0;
		Integer negativeCt = 0;
		if(data.attribute(bestSplit).isNumeric())
		{
			for(Integer i : records)
			{
				if(lessThan && (data.instance(i).value(bestSplit) < (Double)featureValue))
				{
					  split.add(i);
					  if(data.instance(i).stringValue(data.classAttribute()).equals("positive"))
						  positiveCt++;
					  else
						  negativeCt++;
				}
				else if ((!lessThan) && (data.instance(i).value(bestSplit) > (Double)featureValue))
				{
				    split.add(i);
				    if(data.instance(i).stringValue(data.classAttribute()).equals("positive"))
						  positiveCt++;
					  else
						  negativeCt++;
				}
			}
		}
		else
		{
			for(Integer i : records)
			{
				if(data.instance(i).toString(bestSplit).equals(featureValue))
				{
				  split.add(i);
				  if(data.instance(i).stringValue(data.classAttribute()).equals("positive"))
					  positiveCt++;
				  else
					  negativeCt++;
				}
			}
		}
		classCts.setNegativeCt(negativeCt);
		classCts.setPositiveCt(positiveCt);
		return split;
	}

	private static List<Object> findFeatureRange(List<Integer> records, Instances data, Integer split) {
		List<Object> featureRange =  new ArrayList<>();
		Enumeration e = data.attribute(split).enumerateValues();
		while(e.hasMoreElements())
			featureRange.add(e.nextElement());
		return featureRange;
	}

	public static void printTree(TreeNode root, int level, Instances data, boolean first)
	{
		if(!(root.getChildren().size() == 0))
		{
			if(level == 0)
			{
				if(data.attribute(root.getSplitIndex()).isNumeric())
				{
					if(!first) System.out.print("\n");
  	    	            System.out.print(data.attribute(root.getSplitIndex()).name()
    		                            + " " + root.getOperator() + " ");
  	    	            System.out.printf("%.6f", root.getSplitValue());
  	    	            System.out.print(" " + "[" + root.getNegativeCount() 
  	    	                           + " " + root.getPositiveCount() + "]");
				}
				else
				{
					if(!first) System.out.print("\n");
					System.out.print(data.attribute(root.getSplitIndex()).name()
                            + " " + root.getOperator() + " " + root.getSplitValue()
                            + " [" + root.getNegativeCount() 
	                           + " " + root.getPositiveCount() + "]");
				}
			}
			else
			{
			    System.out.print("\n");
		  	    for(int i = 0 ; i < level; i++)
		  		    System.out.print("|\t");
		  	    if(data.attribute(root.getSplitIndex()).isNumeric())
			    {
		  	       System.out.print(data.attribute(root.getSplitIndex()).name()
		  	    		           + " " + root.getOperator() + " " );
		  	       System.out.printf("%.6f", root.getSplitValue());
		  	       System.out.print(" " + "[" + root.getNegativeCount() 
                                  + " " + root.getPositiveCount() + "]");
			    }
		  	    else
		  	    {
		  	    	   System.out.print(data.attribute(root.getSplitIndex()).name()
                            + " " + root.getOperator() + " " + root.getSplitValue()
                            + " [" + root.getNegativeCount() 
	                           + " " + root.getPositiveCount() + "]");
		  	    }
			}
	  	    for(TreeNode n : root.getChildren())
	  		    printTree(n,level+1, data, false);
		}
		else System.out.print(": " + root.getSplitValue());
	}
	
	private static boolean isSatisfied(TreeNode treeNode, Instance instance) {
		boolean holds = false;
		Integer nodeIndex = treeNode.getSplitIndex();
		String operator = treeNode.getOperator();
		Object nodeValue = treeNode.getSplitValue();
		if(operator == null) return true;
			
		if("<=".equals(operator))
		{
			Double lhs = (Double)nodeValue;
			holds = (instance.value(nodeIndex) <= lhs); 
		}
		else if(">".equals(operator))
		{
			Double lhs = (Double)nodeValue;
			holds = (instance.value(nodeIndex) > lhs); 
		}
		else if("=".equals(operator))
		{
			String lhs = (String)nodeValue;
			holds = lhs.equals(instance.stringValue(nodeIndex));
		}
		return holds;
	}
	
	private static String findPrediction(TreeNode root, Instance instance) {
		String prediction = null;
		if(root.getChildren().size() == 0)
		{
			prediction = (String)root.getSplitValue();
		}
		else
		{
         	for(int i = 0; i < root.getChildren().size();i++)
         	{
         		if(isSatisfied(root.getChildren().get(i), instance))
         		{
         			prediction = findPrediction(root.getChildren().get(i), instance);
         			break;
         		}
         	}
		}
		return prediction;
	}

	public static void main(String args[])
	{
	    BufferedReader reader;
	    BufferedReader readerTest;
	    try {
	      reader = new BufferedReader(new FileReader(args[0]));
	    	  //reader = new BufferedReader(new FileReader("diabetes_train.arff"));
		  Instances data = new Instances(reader);
		  data.setClassIndex(data.numAttributes() - 1);
		  reader.close();
		  List<Integer> records = new ArrayList<Integer>();
		  for(int i = 0; i < data.numInstances();i++)
		    records.add(i);
		  TreeNode root = new TreeNode();
		  root.setParent(null);
          //root.setValue("Start");
		  findDecisionTree(data, records, root, Integer.parseInt(args[2]));
          //findDecisionTree(data, records, root, 2);
          for(int i = 0; i < root.getChildren().size(); i++)
          {
        	     boolean first = false;
        	     if(i == 0) first = true;
        	     printTree(root.getChildren().get(i),0, data,first);
          }
          readerTest = new BufferedReader(new FileReader(args[1]));
          //readerTest = new BufferedReader(new FileReader("diabetes_test.arff"));
          Instances test = new Instances(readerTest);
          test.setClassIndex(test.numAttributes()-1);
          readerTest.close();
          int correctPredictCt = 0;
          System.out.println("\n<Predictions for the Test Set Instances>");
          //1: Actual: positive Predicted: positive
          for(int i = 0; i < test.numInstances(); i++)
          {
        	         String prediction = findPrediction(root, test.instance(i));
	        	     if(prediction.equals(test.instance(i).stringValue(test.classAttribute())))
	        	    	   correctPredictCt++;
	        	     System.out.println((i +1) + ": " + "Actual: " + test.instance(i).stringValue(test.classAttribute())
	        	    		                 + " Predicted: " + prediction);
	        	     
          }
          //Number of correctly classified: 68 Total number of test instances: 100
          System.out.println("Number of correctly classified: " + correctPredictCt
        		                + " Total number of test instances: " + test.numInstances());
          //printTree(root,0, data);
		} catch (FileNotFoundException e) {
		  e.printStackTrace();
		} catch (IOException e) {
		  e.printStackTrace();
		}
	}

}
