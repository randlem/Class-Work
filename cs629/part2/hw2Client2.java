import java.net.*;
import java.io.*;
import java.text.*;
import java.util.*;

public class hw2Client2 {

	public static void main(String[] args) {
		String url = "http://www.vogonpoetryreview.com/cs629/";
		String target = "";
		String send_msg = "";
		String recv_msg = "";
		SimpleDateFormat currDate =
			new SimpleDateFormat("yyyyMMdd-HH:mm:ss:SSSS");

		switch (args.length) {
			case 3: {
				url = args[2];
			}
			case 2: {
				target = args[0];
				send_msg = args[1];
			}break;
			default: {
				System.out.println("Usage: hw2Client <target> <message> [url]");
				System.exit(-1);
			}break;
		}

		try {
			URL[] loadUrl = { new URL(url) };
			ClassLoader loader = new URLClassLoader(loadUrl);
			Class cH = loader.loadClass("remoteClass");
			Object rCH = cH.newInstance();

			remoteClassInterface senderClassH = (remoteClassInterface) rCH;

			senderClassH.setClient(target);
			senderClassH.sendMessage(send_msg);
			recv_msg = senderClassH.getMessage();

			System.out.println(recv_msg);

		} catch (ClassNotFoundException e) {
			System.err.println(e);
		} catch (IOException e) {
			System.err.println(e);
		} catch (InstantiationException e) {
			System.err.println(e);
		} catch (IllegalAccessException e) {
			System.err.println(e);
		}
	}

}
