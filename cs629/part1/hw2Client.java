import java.net.*;
import java.io.*;
import java.text.*;
import java.util.*;

public class hw2Client {

	public static void main(String[] args) {
		Socket sock;
		PrintStream sockStreamOut;
		DataInputStream sockStreamIn;
		String host = "localhost";
		int port = 1337;
		String target = "";
		String send_msg = "";
		String recv_msg = "";
		SimpleDateFormat currDate =
			new SimpleDateFormat("yyyyMMdd-HH:mm:ss:SSSS");

		switch (args.length) {
			case 4: {
				port = Integer.parseInt(args[3]);
			}
			case 3: {
				host = args[2];
			}
			case 2: {
				target = args[0];
				send_msg = args[1];
			}break;
			default: {
				System.out.println("Usage: hw2Client <target> <message>");
				System.exit(-1);
			}break;
		}

		try {
			sock = new Socket(host,port);
			sockStreamIn = new DataInputStream(sock.getInputStream());	
			sockStreamOut = new PrintStream(sock.getOutputStream(),true);

			sockStreamOut.println(target + "|" + currDate.format(new Date())
				+ "|" + send_msg);
	
			recv_msg = sockStreamIn.readLine();
			System.out.println(recv_msg);

			sockStreamOut.println("recv");

			sockStreamIn.close();
			sockStreamOut.close();
			sock.close();

		} catch (UnknownHostException e) {
			System.err.println(e);
		} catch (IOException e) {
			System.err.println(e);
		}
	}

}
