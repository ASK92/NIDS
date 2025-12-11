Generate a synthetic network flow dataset for Network Intrusion Detection System (NIDS) testing. Create a CSV file with the following specifications:

**Dataset Requirements:**
- Format: CSV file
- Total samples: 10,000 rows (8,000 benign, 2,000 attack)
- Include a 'Label' column: 0 for benign, 1 for attack
- Include an 'Attack' column: 'Benign' for normal traffic, attack type names for malicious traffic

**Required Features (columns):**

1. **L4_SRC_PORT** - Source port number (integer, range: 1024-65535)
2. **L4_DST_PORT** - Destination port number (integer, common ports: 80, 443, 22, 21, 25, 53, 3389, etc.)
3. **PROTOCOL** - IP protocol number (integer: 6=TCP, 17=UDP, 1=ICMP)
4. **L7_PROTO** - Application layer protocol (string: 'HTTP', 'HTTPS', 'SSH', 'FTP', 'DNS', 'SMTP', 'RDP', 'Unknown')
5. **IN_BYTES** - Incoming bytes (integer, range: 0-10,000,000)
6. **OUT_BYTES** - Outgoing bytes (integer, range: 0-10,000,000)
7. **IN_PKTS** - Incoming packets (integer, range: 1-50,000)
8. **OUT_PKTS** - Outgoing packets (integer, range: 1-50,000)
9. **TCP_FLAGS** - TCP flags (integer, common: 2=SYN, 18=SYN-ACK, 16=ACK, 25=FIN-ACK-PSH)
10. **CLIENT_TCP_FLAGS** - Client TCP flags (integer, range: 0-255)
11. **SERVER_TCP_FLAGS** - Server TCP flags (integer, range: 0-255)
12. **FLOW_DURATION_MILLISECONDS** - Flow duration in milliseconds (integer, range: 1-3,600,000)
13. **DURATION_IN** - Duration of incoming flow (integer, milliseconds)
14. **DURATION_OUT** - Duration of outgoing flow (integer, milliseconds)
15. **MIN_TTL** - Minimum TTL value (integer, range: 1-255)
16. **MAX_TTL** - Maximum TTL value (integer, range: 1-255)
17. **LONGEST_FLOW_PKT** - Longest packet in flow (integer, bytes, range: 64-1514)
18. **SHORTEST_FLOW_PKT** - Shortest packet in flow (integer, bytes, range: 64-1514)
19. **MIN_IP_PKT_LEN** - Minimum IP packet length (integer, range: 20-1500)
20. **MAX_IP_PKT_LEN** - Maximum IP packet length (integer, range: 20-1500)
21. **SRC_TO_DST_SECOND_BYTES** - Bytes per second from source to destination (float)
22. **DST_TO_SRC_SECOND_BYTES** - Bytes per second from destination to source (float)
23. **RETRANSMITTED_IN_BYTES** - Retransmitted incoming bytes (integer, range: 0-1,000,000)
24. **RETRANSMITTED_IN_PKTS** - Retransmitted incoming packets (integer, range: 0-10,000)
25. **RETRANSMITTED_OUT_BYTES** - Retransmitted outgoing bytes (integer, range: 0-1,000,000)
26. **RETRANSMITTED_OUT_PKTS** - Retransmitted outgoing packets (integer, range: 0-10,000)
27. **SRC_TO_DST_AVG_THROUGHPUT** - Average throughput source to destination (float, bytes/second)
28. **DST_TO_SRC_AVG_THROUGHPUT** - Average throughput destination to source (float, bytes/second)
29. **NUM_PKTS_UP_TO_128_BYTES** - Packets up to 128 bytes (integer, range: 0-10,000)
30. **NUM_PKTS_128_TO_256_BYTES** - Packets 128-256 bytes (integer, range: 0-10,000)
31. **NUM_PKTS_256_TO_512_BYTES** - Packets 256-512 bytes (integer, range: 0-10,000)
32. **NUM_PKTS_512_TO_1024_BYTES** - Packets 512-1024 bytes (integer, range: 0-10,000)
33. **NUM_PKTS_1024_TO_1514_BYTES** - Packets 1024-1514 bytes (integer, range: 0-10,000)
34. **TCP_WIN_MAX_IN** - Maximum TCP window size incoming (integer, range: 0-65535)
35. **TCP_WIN_MAX_OUT** - Maximum TCP window size outgoing (integer, range: 0-65535)
36. **ICMP_TYPE** - ICMP type (integer, range: 0-255, mostly 0 or 8 for ping)
37. **ICMP_IPV4_TYPE** - ICMP IPv4 type (integer, range: 0-255)
38. **DNS_QUERY_ID** - DNS query ID (integer, range: 0-65535)
39. **DNS_QUERY_TYPE** - DNS query type (integer: 1=A, 2=NS, 5=CNAME, 15=MX, 28=AAAA)
40. **DNS_TTL_ANSWER** - DNS TTL in answer (integer, range: 0-86400)
41. **FTP_COMMAND_RET_CODE** - FTP command return code (integer: 200, 220, 230, 331, 530, etc.)

**Data Characteristics:**

**Benign Traffic (80% of data, Label=0, Attack='Benign'):**
- Normal web browsing: HTTP/HTTPS traffic, ports 80/443, moderate bytes/packets
- Email: SMTP traffic, port 25, small-medium packets
- File transfer: FTP traffic, port 21, larger bytes
- DNS queries: UDP port 53, small packets
- SSH sessions: Port 22, moderate traffic
- Normal TCP connections with proper handshakes

**Attack Traffic (20% of data, Label=1):**
- DDoS attacks: Very high packet counts, short duration, many SYN flags
- Port scanning: Many connections to different ports, short duration
- Brute force: Many failed connection attempts, high retransmissions
- Data exfiltration: Unusually high OUT_BYTES, long duration
- Malicious payloads: Large packet sizes, unusual protocols

**Output Format:**
Provide the complete CSV file with:
1. Header row with all column names
2. 10,000 data rows
3. Realistic correlations between features (e.g., high IN_BYTES should correlate with high IN_PKTS)
4. Some missing values (NaN) in optional fields like DNS_QUERY_ID, FTP_COMMAND_RET_CODE (about 5-10% of rows)

**Example row format:**
L4_SRC_PORT,L4_DST_PORT,PROTOCOL,L7_PROTO,IN_BYTES,OUT_BYTES,IN_PKTS,OUT_PKTS,TCP_FLAGS,CLIENT_TCP_FLAGS,SERVER_TCP_FLAGS,FLOW_DURATION_MILLISECONDS,DURATION_IN,DURATION_OUT,MIN_TTL,MAX_TTL,LONGEST_FLOW_PKT,SHORTEST_FLOW_PKT,MIN_IP_PKT_LEN,MAX_IP_PKT_LEN,SRC_TO_DST_SECOND_BYTES,DST_TO_SRC_SECOND_BYTES,RETRANSMITTED_IN_BYTES,RETRANSMITTED_IN_PKTS,RETRANSMITTED_OUT_BYTES,RETRANSMITTED_OUT_PKTS,SRC_TO_DST_AVG_THROUGHPUT,DST_TO_SRC_AVG_THROUGHPUT,NUM_PKTS_UP_TO_128_BYTES,NUM_PKTS_128_TO_256_BYTES,NUM_PKTS_256_TO_512_BYTES,NUM_PKTS_512_TO_1024_BYTES,NUM_PKTS_1024_TO_1514_BYTES,TCP_WIN_MAX_IN,TCP_WIN_MAX_OUT,ICMP_TYPE,ICMP_IPV4_TYPE,DNS_QUERY_ID,DNS_QUERY_TYPE,DNS_TTL_ANSWER,FTP_COMMAND_RET_CODE,Label,Attack

Generate realistic network flow data that could be used to test a machine learning-based intrusion detection system.





