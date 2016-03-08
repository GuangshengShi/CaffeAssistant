package caffe.parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * 
 * @author thesby
 *
 */
public class Parser {
	private ArrayList<StringBuffer> net;
	private StringBuffer netInfo;  //TODO: show the structure of the net.
	private StringBuffer solver;
	private ArrayList<StringBuffer> process;
	
	private String logID; 
	
	private BufferedReader br;

	/**
	 * This is the start of a parser.
	 * @param logFileName
	 * The name of text file that you want to parse
	 */
	public Parser(String logFileName) {
		initialUtil(logFileName);
		parse(false);
		System.out.println("Done");
	}
	
	/**
	 * This is the start of a parser.
	 * @param logFileName
	 * The name of text file that you want to parse
	 * @param ifPrefix
	 * true: You want the prefix of each column, "Iter", "loss", "lr". But it's not 
	 * good for matlab or python to plot a figure.  false: No prefix
	 */
	public Parser(String logFileName, boolean ifPrefix) {
		initialUtil(logFileName);
		parse(ifPrefix);
		//System.out.println("Done");
	}
	
	/**
	 * Initialize the utilization
	 * @param logFileName
	 * File name of the log file.
	 */
	private void initialUtil(String logFileName){
		try {
			br = new BufferedReader(new FileReader(logFileName));
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + logFileName);
			System.exit(-1);
		}
		//initialize the util
		net = new ArrayList<>();
		netInfo = new StringBuffer();
		solver = new StringBuffer();
		process = new ArrayList<>();
	}
	
	/**
	 * main function to parse the log file.
	 * @param ifPrefix
	 * Control the prefix
	 */
	private void parse(boolean ifPrefix){
		boolean prefix = ifPrefix;
		String line = "";
		try {
			//parse solver
			boolean isSolverStart = false;
			boolean isSolverEnd = false;
			while((line = br.readLine()) != null){
				line = line.trim();
				//System.out.println(line);
				String []words = line.split(" ");
				for(int i = 0; i < words.length - 1; ++i){
					//如果行内包含了"Using" "GPUs",则认为从此开始解析
					if(words[i].equals("Using") && words[i + 1].equals("GPUs")){
						isSolverStart = true;
						logID = words[0];
					}else if(words[0].equals("net:")){
						isSolverEnd = true;
					}
				}
				if(isSolverStart){
					solver.append(line + '\n');
				}
				if(isSolverEnd){
					break;
				}
			} //while
			
			//parse net and net info
			StringBuffer trainNet = new StringBuffer();
			StringBuffer testNet = new StringBuffer();
			boolean isTrainNetStart = false;
			boolean isTrainNetEnd = false;
			while((line = br.readLine()) != null){
				String line2 = line + "";
				line = line.trim();
				String []words = line.split(" ");
				if(words[0].equals("name:"))
					isTrainNetStart = true;
				if(words[0].equals(logID))
					isTrainNetEnd = true;
				else
					isTrainNetEnd = false;
				if(isTrainNetStart && !isTrainNetEnd){
					trainNet.append(line2 + '\n');
				}
				if(isTrainNetStart && isTrainNetEnd)
					break;
			}

			boolean isTestNetStart = false;
			boolean isTestNetEnd = false;
			while((line = br.readLine()) != null){
				String line2 = line + "";
				line = line.trim();
				String []words = line.split(" ");
				if(words[0].equals("name:"))
					isTestNetStart = true;
				if(words[0].equals(logID))
					isTestNetEnd = true;
				else 
					isTestNetEnd = false;
				if(isTestNetStart && !isTestNetEnd){
					testNet.append(line2 + '\n');
				}
				if(isTestNetStart && isTestNetEnd)
					break;
			}
			net.add(trainNet);
			net.add(testNet);
			
			//process
			StringBuffer trainProc = new StringBuffer();
			StringBuffer testProc = new StringBuffer();
			while((line = br.readLine()) != null){
				line = line.trim();
				String []words = line.split(" ");
				if(line.contains("Iteration") && line.contains("loss")){ //Train info
					String trainProcStr = "";
					for(int i = 0; i < words.length; ++i){
						if(words[i].equals("Iteration"))
							trainProcStr += (prefix?"Iter=":"")+words[i + 1].replace(',', ' ');
						if(words[i].equals("loss") && words[i + 1].equals("="))
							trainProcStr += (prefix?"loss=":"")+words[i + 2] + " ";
					}
					line = br.readLine(); //这一行是用于解析不同输出,暂未使用,所以略过
					line = br.readLine();
					String []words2 = line.split(" ");
					for(int i = 0; i < words2.length; ++i){
						if(words2[i].equals("lr") && words2[i + 1].equals("="))
							trainProcStr += (prefix?"lr=":"") + words2[i + 2];
					}
					trainProc.append(trainProcStr + '\n');
				}else if(line.contains("Iteration") && line.contains("Testing")){//Test
					String testProcStr = "";
					for(int i = 0; i < words.length; ++i){
						if(words[i].equals("Iteration"))
							testProcStr += (prefix?"Iter=":"") + words[i + 1].replace(',', ' ');
					}
					line = br.readLine(); //accuracy
					String []words2 = line.split(" ");
					for(int i = 0; i < words2.length; ++i){
						if(words2[i].equals("accuracy") && words2[i + 1].equals("=")){
							testProcStr += (prefix?"acc=":"") + words2[i + 2] + " ";
						}
					}
					line = br.readLine(); //loss
					String []words3 = line.split(" ");
					for(int i = 0; i < words3.length; ++i){
						if(words3[i].equals("loss") && words3[i + 1].equals("=")){
							testProcStr += (prefix?"loss=":"") + words3[i + 2];
						}
					}
					testProc.append(testProcStr + '\n');
				}
			}
			process.add(trainProc);
			process.add(testProc);
		} catch (IOException e) {
			System.out.println("Error: can not read file!");
		}
	}
	
	/**
	 * Save the parsing result to files
	 * @param prefix
	 * the prefix you want to add, for example, a path prefix.
	 */
	public void saveToFile(String prefix){
		
		//保存solver
		try {
			BufferedWriter bWriter = new BufferedWriter(
					new FileWriter(new File(prefix + "solver.txt")));
			bWriter.write(solver.toString());
			bWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//Save net
		try {
			BufferedWriter bWriter = new BufferedWriter(
					new FileWriter(new File(prefix + "train_net.txt")));
			bWriter.write(net.get(0).toString());
			bWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			BufferedWriter bWriter = new BufferedWriter(
					new FileWriter(new File(prefix + "test_net.txt")));
			bWriter.write(net.get(1).toString());
			bWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//Save process
		try {
			BufferedWriter bWriter = new BufferedWriter(
					new FileWriter(new File(prefix + "train_process.txt")));
			bWriter.write(process.get(0).toString());
			bWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			BufferedWriter bWriter = new BufferedWriter(
					new FileWriter(new File(prefix + "test_process.txt")));
			bWriter.write(process.get(1).toString());
			bWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
