#ifndef UART_COMMUNICATION_H
#define UART_COMMUNICATION_H

#ifdef __cplusplus
extern "C" {
#endif

void uart_init(void);
void uart_send_data(const char* data);
void uart_receive_data(char* buffer);


#ifdef __cplusplus
}
#endif

#endif // UART_COMMUNICATION_H
