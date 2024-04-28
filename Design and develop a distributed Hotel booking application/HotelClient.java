import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class HotelClient {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        try {
            // Connect to the RMI Registry running on localhost and default RMI port
            Registry registry = LocateRegistry.getRegistry("localhost", 1099);
            Hotel stub = (Hotel) registry.lookup("HotelService");

            while (true) {
                System.out.println("\nChoose an option:");
                System.out.println("1. Book a room");
                System.out.println("2. Cancel a booking");
                System.out.println("3. Exit");
                System.out.print("Enter choice: ");
                int choice = scanner.nextInt();
                scanner.nextLine();  // Consume the newline left-over

                String guestName, response;
                switch (choice) {
                    case 1:  // Booking a room
                        System.out.print("Enter guest name to book: ");
                        guestName = scanner.nextLine();
                        response = stub.bookRoom(guestName);
                        System.out.println("Response: " + response);
                        break;
                    case 2:  // Cancelling a booking
                        System.out.print("Enter guest name to cancel booking: ");
                        guestName = scanner.nextLine();
                        response = stub.cancelBooking(guestName);
                        System.out.println("Response: " + response);
                        break;
                    case 3:  // Exit
                        System.out.println("Exiting...");
                        scanner.close();
                        return;
                    default:
                        System.out.println("Invalid choice, please enter 1, 2, or 3.");
                }
            }
        } catch (Exception e) {
            System.err.println("Client exception: " + e.toString());
            e.printStackTrace();
        } finally {
            scanner.close();
        }
    }
}
