use crate::{Request, Response};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub(crate) fn read_request(reader: &mut impl Read) -> Result<Request, bincode::Error> {
    bincode::deserialize_from::<_, Request>(reader)
}

pub(crate) fn write_request(
    writer: &mut impl Write,
    request: &Request,
) -> Result<(), bincode::Error> {
    bincode::serialize_into(writer, request)
}

#[derive(Serialize, Deserialize)]
struct ResponseHeader {
    id: usize,
    len: usize,
}

pub(crate) fn read_response(reader: &mut impl Read) -> Result<Response, bincode::Error> {
    let header = bincode::deserialize_from::<_, ResponseHeader>(&mut *reader)?;
    let id = header.id;
    let len = header.len;
    if len != 0 {
        let mut data = vec![0u8; len].into_boxed_slice();
        if let Err(e) = reader.read_exact(&mut data[..]) {
            Err(bincode::Error::from(e))
        } else {
            Ok(Response {
                id,
                data: Some(data),
            })
        }
    } else {
        Ok(Response { id, data: None })
    }
}

pub(crate) fn write_response(
    writer: &mut impl Write,
    id: usize,
    data: Option<&[u8]>,
) -> Result<(), bincode::Error> {
    let len = match data {
        Some(data) => data.len(),
        None => 0,
    };
    let header = ResponseHeader { id, len };
    if let Err(e) = bincode::serialize_into(&mut *writer, &header) {
        return Err(e);
    }
    // has payload
    if len != 0 {
        if let Err(e) = writer.write_all(data.unwrap()) {
            return Err(bincode::Error::from(e));
        }
    }
    Ok(())
}
