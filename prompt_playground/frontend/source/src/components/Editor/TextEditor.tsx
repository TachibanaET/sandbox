import React from "react";
import Form from "react-bootstrap/Form";

type Props = {
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const TextEditor: React.FC<Props> = (props) => {
    return (
        <Form>
            <Form.Group className="mb-3" controlId="pg-prompt">
                <Form.Control
                    as="textarea" rows={10} placeholder="Try to write sth."
                    onChange={props.onChange} />
            </Form.Group>
        </Form>
    )
}

export default TextEditor;