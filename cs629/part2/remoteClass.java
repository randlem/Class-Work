import java.net.*;
import java.io.*;
import java.text.*;
import java.util.*;

public class remoteClass implements remoteClassInterface {

	public remoteClass() {
		openSocket();
	}

	public void setClient(String c) {
		client = c;
	}

	public void openSocket() {
		try {
			sock = new Socket(server,port);
			sockStreamIn = new DataInputStream(sock.getInputStream());
			sockStreamOut = new PrintStream(sock.getOutputStream(),true);
		} catch (UnknownHostException e) {
			System.err.println(e);
		} catch (IOException e) {
			System.err.println(e);
		}
	}

	public void closeSocket() {
		try {
			sockStreamOut.close();
			sockStreamIn.close();
			sock.close();
		} catch (UnknownHostException e) {
			System.err.println(e);
		} catch (IOException e) {
			System.err.println(e);
		}
	}

	public void sendMessage(String msg) {
		SimpleDateFormat currDate =
			new SimpleDateFormat("yyyyMMdd-HH:mm:ss:SSSS");

		sockStreamOut.println(client + "|" + currDate.format(new Date())
			+ "|" + msg);
	}

	public String getMessage() {
		String msg = "";

		try {
			msg = sockStreamIn.readLine();
			sockStreamOut.println("recv");
			closeSocket();
		} catch (UnknownHostException e) {
			System.err.println(e);
		} catch (IOException e) {
			System.err.println(e);
		}

		return msg;
	}

	private Socket sock;
	PrintStream sockStreamOut;
	DataInputStream sockStreamIn;
	private String client;

	public static final String server = "vogonpoetryreview.com";
	public static final int port = 1337;

}