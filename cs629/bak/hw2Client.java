import java.net.*;
import java.io.*;
import java.text.*;
import java.util.*;

public class hw2Client {

	static final String host	= "localhost";
	static final int port		= 1337;

	public static void main(String[] args) {
		Socket sock;
		PrintStream sockStreamOut;
		BufferedReader sockStreamIn;
		String msg = "";
		SimpleDateFormat currDate =
			new SimpleDateFormat("yyyyMMdd-HH:mm:ss:SSSS");

		if (args.length != 2) {
			System.out.println("Usage: hw2Client <target> <message>");
			System.exit(-1);
		}

		try {
			sock = new Socket(host,port);
			sockStreamIn = new BufferedReader(
				new InputStreamReader(sock.getInputStream()));
			sockStreamOut = new PrintStream(sock.getOutputStream(),true);

			sockStreamOut.println(args[0] + "|" + currDate.format(new Date())
				+ "|" + args[1]);

			msg = sockStreamIn.readLine();
			System.out.println(msg);

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
