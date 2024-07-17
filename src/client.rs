use crate::serialization::{read_response, write_request};
use crate::{Operation, Request, Response};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;

pub struct KVClient {
    request_writer: BufWriter<TcpStream>,
    response_reader: BufReader<TcpStream>,
}

impl KVClient {
    pub fn new(host: &str, port: &str) -> Option<Self> {
        let addr: String = "".to_string() + host + ":" + port;
        match TcpStream::connect(&addr) {
            Ok(s) => {
                let s2 = s.try_clone().unwrap_or_else(|e| {
                    panic!("KVClient fails to clone a tcp stream: {}", e);
                });
                Some(KVClient {
                    request_writer: BufWriter::new(s),
                    response_reader: BufReader::new(s2),
                })
            }
            Err(_) => None,
        }
    }

    pub fn send_requests(&mut self, requests: &Vec<Request>) {
        for r in requests {
            assert!(write_request(&mut self.request_writer, r).is_ok())
        }
        assert!(self.request_writer.flush().is_ok());
    }

    // recv all (drain the buffer)
    pub fn recv_responses(&mut self) -> Vec<Response> {
        assert!(self.response_reader.fill_buf().is_ok());
        let mut responses = Vec::new();

        while !self.response_reader.buffer().is_empty() {
            match read_response(&mut self.response_reader) {
                Ok(r) => {
                    responses.push(r);
                }
                Err(e) => {
                    panic!("KVClient failed to read response: {}", e);
                }
            }
        }

        responses
    }

    pub fn recv_responses_n(&mut self, nr: usize) -> Vec<Response> {
        let mut responses = Vec::<Response>::with_capacity(nr);

        for _ in 0..nr {
            match read_response(&mut self.response_reader) {
                Ok(r) => {
                    responses.push(r);
                }
                Err(e) => {
                    panic!("KVClient failed to read response: {}", e);
                }
            }
        }

        responses
    }

    pub fn set(&mut self, key: &[u8], value: &[u8]) {
        let mut requests = Vec::<Request>::with_capacity(1);
        let op = Operation::Set {
            key: key.into(),
            value: value.into(),
        };
        requests.push(Request { id: 0, op });
        self.send_requests(&requests);

        let mut responses = self.recv_responses_n(1);
        assert_eq!(responses.len(), 1);
        let response = responses.pop().unwrap();
        assert_eq!(response.id, 0);
        assert!(response.data.is_none());
    }

    pub fn get(&mut self, key: &[u8]) -> Option<Box<[u8]>> {
        let mut requests = Vec::<Request>::with_capacity(1);
        let op = Operation::Get { key: key.into() };
        requests.push(Request { id: 0, op });
        self.send_requests(&requests);

        let mut responses = self.recv_responses_n(1);
        assert_eq!(responses.len(), 1);
        let response = responses.pop().unwrap();
        assert_eq!(response.id, 0);
        match response.data {
            Some(v) => Some(v),
            None => None,
        }
    }
}
