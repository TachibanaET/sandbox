import React from "react";
import {
    Button, Form
} from "react-bootstrap";

const Editor: React.FC = () => {
    const handleSubmit = ((e: React.MouseEvent<HTMLElement>) => {
        console.log(e);
    });
    return (
        <>
            <Form>
                <Form.Group className="mb-3" controlId="pg-prompt">
                    <Form.Control as="textarea" rows={3} placeholder="Try to write sth." />
                </Form.Group>
            </Form>
            <Button variant="outline-primary" type="submit" id="submit-prompt" onClick={handleSubmit}>
                Submit
            </Button>
        </>
    )
}

export default Editor;